#!/usr/bin/env python3
"""
Continuous (method x token x layer) sweep of persona-vector extraction (issue #263).

Extends `scripts/extract_persona_vectors.py`'s per-token hooking pattern to dump
activations at every (last-K input position, every response-token-index in the
geometric grid).

Methods supported (`--methods`):
- `a`         : Method A — last-input-token hidden state at i=-1 (legacy reference).
- `a_per_token` : Same forward pass as A; dumps i ∈ {-5..-1} from --prompt-token-positions.
                  Counted in A's forward-pass budget.
- `b`         : Method B — mean over greedily-generated response tokens (Chen 2025).
                  Reuses #218's response cache iff `--reuse-cache` is set + cache passes
                  the Stage 0b shape assertion.
- `bstar`     : Method B* — same as B but excluding the final response token.
- `c1`,`c2`,`c3` : Method C variants from #201.
- `r_per_token`: Per-generation-token hidden states at t ∈ {0,1,2,4,8,16,32,64,128}.
- `caa`       : (descriptive only; per plan §3 v3 fix 1) — invokes `run_caa.py` logic.
                Excluded from the H2 argmax candidate set in the analysis script.

Output layout
-------------
Centroids:
  <output-dir>/method_<m>__pos_<p>__layer_<l>/<role>.pt   shape (D,) fp32
Per-question caches (only when relevant — methods a, b, b*, r_per_token):
  <output-dir>/method_<m>/<role>__per_q.pt                shape (n_q, n_layers, D) fp16

Stage 0b cache-shape assertion
------------------------------
Before any heavy work, if `--reuse-cache <path>` is provided, this script asserts:
    torch.load(<path>/method_a/<role>__per_q.pt).shape == (n_questions, n_layers, hidden_dim)
on a sample of 5 random roles. If the assertion fails, the script EXITS 1 with a
pointer to the §10 fallback. (Per plan §10.)

Usage
-----
Smoke (CPU, ~5 min on local VM — Stage 0a):
  uv run python scripts/sweep_extraction_grid.py \\
      --n-personas 2 --n-questions 4 --layers 0,7 \\
      --prompt-token-positions -1,-2 --response-token-positions 0,1 \\
      --methods a,caa --output-dir /tmp/issue_263_smoke --device cpu

Full Stage 1 sweep (1x H100, ~3.5 GPU-hr):
  nohup uv run python scripts/sweep_extraction_grid.py \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27 \\
      --prompt-token-positions -5,-4,-3,-2,-1 \\
      --response-token-positions 0,1,2,4,8,16,32,64,128 \\
      --methods a,b,bstar,c1,c2,c3,caa \\
      --n-prompts 1 --n-questions 240 \\
      --reuse-cache data/persona_vectors/issue_218/qwen2.5-7b-instruct \\
      --output-dir data/persona_vectors/issue_263/qwen2.5-7b-instruct \\
      --gpu-id 0 --seed 42 > /workspace/logs/issue_263_sweep.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = Path(__file__).parent.parent / "data" / "assistant_axis"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LAYERS_FULL = list(range(28))
DEFAULT_PROMPT_POSITIONS = [-5, -4, -3, -2, -1]
DEFAULT_RESPONSE_POSITIONS = [0, 1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_METHODS = ["a", "b", "bstar", "c1", "c2", "c3", "r_per_token", "caa"]
DEFAULT_HIDDEN_DIM = 3584  # Qwen2.5-7B-Instruct
DEFAULT_N_LAYERS = 28
DEFAULT_MAX_TOKENS = 200  # Method B / R_per_token greedy generation


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_roles(roles_filter: list[str] | None = None) -> dict[str, list[str]]:
    """Load role -> list of system prompts from instruction files."""
    role_list_path = DATA_DIR / "role_list.json"
    instructions_dir = DATA_DIR / "instructions"
    with open(role_list_path) as f:
        all_roles = json.load(f)
    if roles_filter:
        all_roles = {k: v for k, v in all_roles.items() if k in roles_filter}
    role_prompts: dict[str, list[str]] = {}
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
    questions: list[str] = []
    with open(questions_path) as f:
        for line in f:
            item = json.loads(line)
            questions.append(item["question"])
    if n_questions:
        questions = questions[:n_questions]
    return questions


# ── Reproducibility metadata ─────────────────────────────────────────────────


def _git_commit_hash() -> str:
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


def _build_run_metadata(args: argparse.Namespace, n_roles: int, n_questions: int) -> dict:
    return {
        "issue": 263,
        "git_commit": _git_commit_hash(),
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "model": args.model,
        "device": args.device,
        "seed": args.seed,
        "n_roles": n_roles,
        "n_questions": n_questions,
        "n_prompts": args.n_prompts,
        "layers": parse_int_list(args.layers),
        "prompt_token_positions": parse_int_list(args.prompt_token_positions),
        "response_token_positions": parse_int_list(args.response_token_positions),
        "methods": [m.strip() for m in args.methods.split(",")],
        "max_new_tokens": args.max_new_tokens,
        "reuse_cache": args.reuse_cache,
    }


# ── Cache-shape assertion (Stage 0b) ─────────────────────────────────────────


def assert_cache_shape(
    reuse_cache_root: Path,
    method: str,
    expected_n_q: int,
    expected_n_layers: int,
    expected_hidden_dim: int,
    sample_size: int = 5,
    expected_n_positions: int | None = None,
) -> None:
    """Hard-fail if any sampled per_q cache doesn't match expected shape.

    Round 2 / B1 fix: the canonical Method A per-q cache is now 4-D
        (n_q, n_layers, n_prompt_positions, D)
    so H2 can evaluate each (i, l) candidate cell in its own activation space.
    Legacy #218 caches are 3-D (n_q, n_layers, D) at i=-1 only — this is detected
    and the §10 fallback path (regenerate Method A through THIS script) is invoked.

    Per plan §10 fallback: if shape is wrong (e.g. (240, 4, 3584) from a
    DEFAULT_LAYERS = [7, 14, 21, 27] launch), exit with a pointer to the
    `--methods a --layers 0..27 --prompt-token-positions -5..-1` regen path.
    """
    method_dir = reuse_cache_root / f"method_{method}"
    if not method_dir.exists():
        raise FileNotFoundError(
            f"Reuse-cache method dir does not exist: {method_dir}\n"
            f"  Pass --reuse-cache <root> where <root>/method_{method}/<role>__per_q.pt exist."
        )
    candidates = sorted(method_dir.glob("*__per_q.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No __per_q.pt files in {method_dir}.\n"
            f"  Cache layout expected: <root>/method_{method}/<role>__per_q.pt"
        )
    rng = random.Random(42)
    sample = rng.sample(candidates, k=min(sample_size, len(candidates)))
    if expected_n_positions is not None:
        expected_shape_4d = (
            expected_n_q,
            expected_n_layers,
            expected_n_positions,
            expected_hidden_dim,
        )
        expected_descr = f"{expected_shape_4d} (4-D, round 2)"
    else:
        expected_shape_4d = None
        expected_descr = (
            f"({expected_n_q}, {expected_n_layers}, {expected_hidden_dim}) [3-D legacy]"
        )
    print(f"Cache-shape assertion (method={method}, expected={expected_descr})")
    for path in sample:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        actual = tuple(tensor.shape)
        # Accept both 4-D (round 2 canonical) and 3-D (legacy #218) layouts:
        ok = False
        if (expected_shape_4d is not None and actual == expected_shape_4d) or actual == (
            expected_n_q,
            expected_n_layers,
            expected_hidden_dim,
        ):
            ok = True
        if not ok:
            msg = (
                f"BAD CACHE SHAPE for {path}: got {actual}, expected {expected_descr}.\n\n"
                "FALLBACK (per plan §10):\n"
                "  Regenerate Method A from scratch through THIS script with the per-token "
                "positions H2 needs:\n"
                "    nohup uv run python scripts/sweep_extraction_grid.py \\\n"
                "        --methods a --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,"
                "18,19,20,21,22,23,24,25,26,27 \\\n"
                "        --prompt-token-positions=-5,-4,-3,-2,-1 \\\n"
                "        --output-dir data/persona_vectors/issue_263/qwen2.5-7b-instruct/"
                "method_a_regen/ &\n"
                "  Then re-run this script with --reuse-cache pointing at the regen dir.\n"
                "  +45 min vs cache-hit branch (accounted in plan §9)."
            )
            print(msg, file=sys.stderr)
            sys.exit(1)
        print(f"  ok: {path.name} -> {actual}", flush=True)
    print("cache-shape OK\n", flush=True)


# ── Chat Template ────────────────────────────────────────────────────────────


def build_chat_text(tokenizer, system_prompt: str, question: str) -> str:
    """Build chat text with system + user messages and add_generation_prompt."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ── Hook helper ──────────────────────────────────────────────────────────────


class HiddenStateCapture:
    """Context manager for capturing hidden states at requested layers via forward hooks."""

    def __init__(self, model, layers: list[int]):
        self.model = model
        self.layers = list(layers)
        self.captured: dict[int, torch.Tensor] = {}
        self._hooks: list = []

    def __enter__(self):
        for layer_idx in self.layers:
            self._hooks.append(
                self.model.model.layers[layer_idx].register_forward_hook(self._make_hook(layer_idx))
            )
        return self

    def __exit__(self, *exc):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        return False

    def _make_hook(self, layer_idx: int):
        def hook_fn(_module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            self.captured[layer_idx] = hs.detach()

        return hook_fn


# ── Method A + A_per_token (prompt-side per-token sweep) ─────────────────────


def extract_prompt_side_grid(  # noqa: C901
    model,
    tokenizer,
    role_prompts: dict[str, list[str]],
    questions: list[str],
    layers: list[int],
    prompt_positions: list[int],
    output_dir: Path,
    n_prompts: int = 1,
    save_per_q: bool = True,
    train_qid_set: set[int] | None = None,
) -> dict[str, dict[tuple[int, int], torch.Tensor]]:
    """Run one forward pass per (role, prompt, question); dump activations at every
    (layer, prompt_position) cell.

    Per plan §4 pseudocode: this is the workhorse for Method A + A_per_token. The
    `i = -1` slice IS Method A; the i ∈ {-5..-2} slices are the new A_per_token.

    Per-question caches (round 2 / B1 + C4 fix):
        method_a/<role>__per_q.pt   shape (n_q, n_layers, n_prompt_positions, D) fp16
    The 4-D layout is REQUIRED for H2 to evaluate each (i, l) candidate cell in its
    own activation space (round-1 BLOCKER B1). Each per-q row carries hidden states
    at every prompt position so the analyzer can slice the right (i, l) tile per cell.

    Train-only centroids (round 2 / B2 fix):
        method_a/<role>__centroid_train.pt   shape (n_layers, n_prompt_positions, D) fp32
    These are means over the train-split questions only (q_idx 0..199 by default), so
    H1 clustering can evaluate cells without consuming the test split. Pass
    `train_qid_set` (a set of question indices) to enable this output. The legacy
    full-question centroid files are still written per cell as before.

    Returns: centroids[role][(layer, position)] = (D,) fp32 tensor (full-question mean).
    """
    print(f"\n{'=' * 60}")
    print("Method A / A_per_token: prompt-side per-token sweep")
    print(
        f"  Roles: {len(role_prompts)}, Prompts/role: {n_prompts}, "
        f"Questions: {len(questions)}, Layers: {len(layers)}, Positions: {prompt_positions}"
    )
    print(f"  Total forward passes: {len(role_prompts) * n_prompts * len(questions)}")
    print(f"{'=' * 60}\n")

    # Pre-make output dirs for each (position, layer) cell
    for pos in prompt_positions:
        for lyr in layers:
            (output_dir / f"method_a__pos_{pos}__layer_{lyr}").mkdir(parents=True, exist_ok=True)
    if save_per_q:
        (output_dir / "method_a").mkdir(parents=True, exist_ok=True)

    centroids: dict[str, dict[tuple[int, int], torch.Tensor]] = {}
    sorted_roles = sorted(role_prompts.items())
    t0 = time.time()

    with HiddenStateCapture(model, layers) as cap:
        for role_idx, (role_name, prompts) in enumerate(sorted_roles):
            # Resume support: if all centroid cells AND per-q cache exist, load from disk.
            cell_paths = [
                output_dir / f"method_a__pos_{pos}__layer_{lyr}" / f"{role_name}.pt"
                for pos in prompt_positions
                for lyr in layers
            ]
            per_q_path = output_dir / "method_a" / f"{role_name}__per_q.pt"
            train_centroid_path = output_dir / "method_a" / f"{role_name}__centroid_train.pt"
            cells_present = all(p.exists() for p in cell_paths)
            per_q_present = (not save_per_q) or per_q_path.exists()
            train_centroid_present = (
                (not save_per_q) or (train_qid_set is None) or train_centroid_path.exists()
            )
            if cells_present and per_q_present and train_centroid_present:
                cached: dict[tuple[int, int], torch.Tensor] = {}
                for pos in prompt_positions:
                    for lyr in layers:
                        cached[(lyr, pos)] = torch.load(
                            output_dir / f"method_a__pos_{pos}__layer_{lyr}" / f"{role_name}.pt",
                            weights_only=True,
                        )
                centroids[role_name] = cached
                print(
                    f"  [{role_idx + 1}/{len(sorted_roles)}] {role_name} — loaded from cache",
                    flush=True,
                )
                continue

            prompts_to_use = prompts[:n_prompts]
            # accum[(lyr, pos)] -> list of (prompt, question) vecs (full set; for centroid)
            accum: dict[tuple[int, int], list[torch.Tensor]] = {
                (lyr, pos): [] for lyr in layers for pos in prompt_positions
            }
            # Train-split accumulators (subset of `accum` indices) for B2 fix.
            # Indexed by the SAME (lyr, pos) keys; only train_qid_set rows are appended.
            train_accum: dict[tuple[int, int], list[torch.Tensor]] | None = (
                {(lyr, pos): [] for lyr in layers for pos in prompt_positions}
                if (train_qid_set is not None)
                else None
            )
            # Per-question 4-D cache: (n_q, n_layers, n_prompt_positions, D) fp16.
            # Round-2 B1 fix: store every prompt position so H2 can evaluate each
            # candidate cell in its own activation space.
            per_q_buffer: list[torch.Tensor] | None = [] if save_per_q else None

            for sys_prompt in prompts_to_use:
                for q_idx, question in enumerate(questions):
                    text = build_chat_text(tokenizer, sys_prompt, question)
                    inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)
                    with torch.no_grad():
                        _ = model(**inputs)
                    seq_len = inputs["input_ids"].shape[1]

                    is_train_q = (train_qid_set is not None) and (q_idx in train_qid_set)

                    # Per-q row across layers AND positions: (n_layers, n_positions, D), fp16.
                    if per_q_buffer is not None:
                        layer_rows: list[torch.Tensor] = []
                        for lyr in layers:
                            hs = cap.captured[lyr]
                            pos_vecs: list[torch.Tensor] = []
                            for pos in prompt_positions:
                                tok_pos = seq_len + pos
                                if tok_pos < 0:
                                    pos_vecs.append(torch.zeros(hs.shape[-1]))
                                else:
                                    pos_vecs.append(hs[0, tok_pos, :].float().cpu())
                            layer_rows.append(torch.stack(pos_vecs))  # (n_pos, D)
                        per_q_buffer.append(
                            torch.stack(layer_rows).to(torch.float16)
                        )  # (n_layers, n_pos, D)

                    for lyr in layers:
                        hs = cap.captured[lyr]
                        for pos in prompt_positions:
                            tok_pos = seq_len + pos
                            if tok_pos < 0:
                                continue
                            vec = hs[0, tok_pos, :].float().cpu()
                            accum[(lyr, pos)].append(vec)
                            if train_accum is not None and is_train_q:
                                train_accum[(lyr, pos)].append(vec)

            per_role: dict[tuple[int, int], torch.Tensor] = {}
            for (lyr, pos), vecs in accum.items():
                if not vecs:
                    continue
                centroid = torch.stack(vecs).mean(dim=0)
                per_role[(lyr, pos)] = centroid
                torch.save(
                    centroid,
                    output_dir / f"method_a__pos_{pos}__layer_{lyr}" / f"{role_name}.pt",
                )
            centroids[role_name] = per_role

            if per_q_buffer is not None and per_q_buffer:
                # Stack to (n_q, n_layers, n_positions, D) fp16
                per_q_tensor = torch.stack(per_q_buffer)
                torch.save(per_q_tensor, per_q_path)

            # Train-only centroid block: (n_layers, n_positions, D) fp32. B2 fix.
            if train_accum is not None:
                n_layers = len(layers)
                n_pos = len(prompt_positions)
                if per_q_buffer is not None and per_q_buffer:
                    D = per_q_buffer[0].shape[-1]
                    train_block = torch.full((n_layers, n_pos, D), float("nan"))
                else:
                    D = next(iter(per_role.values())).shape[-1] if per_role else 0
                    train_block = torch.full((n_layers, n_pos, D), float("nan")) if D else None
                if train_block is not None:
                    for li, lyr in enumerate(layers):
                        for pi, pos in enumerate(prompt_positions):
                            vecs = train_accum[(lyr, pos)]
                            if vecs:
                                train_block[li, pi, :] = torch.stack(vecs).mean(dim=0)
                    torch.save(train_block.float(), train_centroid_path)

            elapsed = time.time() - t0
            rate = (role_idx + 1) / elapsed * 60 if elapsed > 0 else 0.0
            print(
                f"  [{role_idx + 1}/{len(sorted_roles)}] {role_name} — "
                f"{elapsed:.0f}s elapsed, {rate:.1f} roles/min",
                flush=True,
            )

    return centroids


# ── Method B (response-mean) + B* (response-mean excluding final token) ──────


def generate_responses_vllm(
    model_name: str,
    role_prompts: dict[str, list[str]],
    questions: list[str],
    n_prompts: int,
    gpu_id: int,
    output_path: Path,
    max_new_tokens: int,
    device: str,
) -> dict[str, list[dict]]:
    """Generate greedy responses for all (role, prompt, question) combos via vLLM.

    Cached at output_path (json). On smoke runs (device=cpu), this is bypassed —
    callers must supply pre-generated responses or run with vLLM in a separate step.
    """
    if output_path.exists():
        with open(output_path) as f:
            cached = json.load(f)
        if len(cached) == len(role_prompts):
            print(f"  Loaded cached responses from {output_path}")
            return cached

    if device != "cuda":
        raise RuntimeError(
            "vLLM generation requires CUDA. Provide pre-generated responses at "
            f"{output_path} or rerun with --device cuda."
        )

    from vllm import LLM, SamplingParams

    print(f"  Loading vLLM ({model_name}) for greedy generation...")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    all_convos = []
    all_keys = []
    for role_name, prompts in sorted(role_prompts.items()):
        for p_idx, sys_prompt in enumerate(prompts[:n_prompts]):
            for question in questions:
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": question},
                ]
                all_convos.append(messages)
                all_keys.append((role_name, p_idx, question))

    outputs = llm.chat(all_convos, sampling_params)

    results: dict[str, list[dict]] = {role: [] for role in role_prompts}
    for (role_name, p_idx, question), output in zip(all_keys, outputs, strict=True):
        results[role_name].append(
            {
                "system_prompt": role_prompts[role_name][p_idx],
                "question": question,
                "response": output.outputs[0].text,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f)

    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def extract_response_methods(  # noqa: C901
    model,
    tokenizer,
    responses: dict[str, list[dict]],
    layers: list[int],
    response_positions: list[int],
    output_dir: Path,
    do_b: bool,
    do_bstar: bool,
    do_r_per_token: bool,
    save_per_q: bool = True,
    train_qid_set: set[int] | None = None,
) -> dict[str, dict[str, dict[tuple[int, int], torch.Tensor]]]:
    """Single forward pass per (role, response) yields hidden states at:
    - layer x every response-token-index in `response_positions` (R_per_token)
    - layer x mean over response tokens (Method B)
    - layer x mean over response tokens excluding the last (Method B*)

    Returns: out[method][role][(layer, position)] = (D,) tensor.

    Note: Method B / B* have a single position slot (treated as position=0 for
    file-layout convenience); R_per_token has n_pos x n_layers cells.
    """
    print(f"\n{'=' * 60}")
    print("Methods B / B* / R_per_token: response-side hidden state extraction")
    print(f"  Roles: {len(responses)}, Layers: {len(layers)}")
    if do_r_per_token:
        print(f"  R_per_token positions: {response_positions}")
    print(f"  Methods: B={do_b}, B*={do_bstar}, R_per_token={do_r_per_token}")
    print(f"{'=' * 60}\n")

    # Output dirs
    if do_b:
        for lyr in layers:
            (output_dir / f"method_b__pos_0__layer_{lyr}").mkdir(parents=True, exist_ok=True)
        if save_per_q:
            (output_dir / "method_b").mkdir(parents=True, exist_ok=True)
    if do_bstar:
        for lyr in layers:
            (output_dir / f"method_bstar__pos_0__layer_{lyr}").mkdir(parents=True, exist_ok=True)
        if save_per_q:
            (output_dir / "method_bstar").mkdir(parents=True, exist_ok=True)
    if do_r_per_token:
        for t in response_positions:
            for lyr in layers:
                (output_dir / f"method_r_per_token__pos_{t}__layer_{lyr}").mkdir(
                    parents=True, exist_ok=True
                )
        if save_per_q:
            (output_dir / "method_r_per_token").mkdir(parents=True, exist_ok=True)

    out: dict[str, dict[str, dict[tuple[int, int], torch.Tensor]]] = {
        "b": {},
        "bstar": {},
        "r_per_token": {},
    }
    sorted_roles = sorted(responses.items())
    t0 = time.time()

    with HiddenStateCapture(model, layers) as cap:
        for role_idx, (role_name, items) in enumerate(sorted_roles):
            # Per-role accumulators (full-question, used for centroid)
            b_accum = {lyr: [] for lyr in layers} if do_b else None
            bstar_accum = {lyr: [] for lyr in layers} if do_bstar else None
            r_accum = (
                {(lyr, t): [] for lyr in layers for t in response_positions}
                if do_r_per_token
                else None
            )
            # Train-only accumulators (B2 fix). Indexed identically to the full ones.
            b_train_accum: dict[int, list[torch.Tensor]] | None = (
                {lyr: [] for lyr in layers} if (do_b and train_qid_set is not None) else None
            )
            bstar_train_accum: dict[int, list[torch.Tensor]] | None = (
                {lyr: [] for lyr in layers} if (do_bstar and train_qid_set is not None) else None
            )
            r_train_accum: dict[tuple[int, int], list[torch.Tensor]] | None = (
                {(lyr, t): [] for lyr in layers for t in response_positions}
                if (do_r_per_token and train_qid_set is not None)
                else None
            )
            # Per-q caches:
            #   B:  (n_q, n_layers, D) fp16
            #   B*: (n_q, n_layers, D) fp16 (B1/C4 round-2 fix)
            #   R_per_token: (n_q, n_layers, n_response_positions, D) fp16 (B1/C4 fix)
            b_per_q_buf: list[torch.Tensor] | None = [] if (do_b and save_per_q) else None
            bstar_per_q_buf: list[torch.Tensor] | None = [] if (do_bstar and save_per_q) else None
            r_per_q_buf: list[torch.Tensor] | None = [] if (do_r_per_token and save_per_q) else None

            n_skipped = 0
            for item_idx, item in enumerate(items):
                sys_prompt = item["system_prompt"]
                question = item["question"]
                response = item["response"]
                if not response:
                    n_skipped += 1
                    continue

                # The "question index" carried in the per-q cache is the position in
                # the (q, prompt) flat list used elsewhere (item_idx). For n_prompts=1
                # this is the question index; for n_prompts>1 it's q + n_q * p_idx.
                is_train_q = (train_qid_set is not None) and (item_idx in train_qid_set)

                prompt_messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": question},
                ]
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
                prompt_ids = tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"]
                prompt_len = prompt_ids.shape[1]

                full_messages = [*prompt_messages, {"role": "assistant", "content": response}]
                full_text = tokenizer.apply_chat_template(
                    full_messages, tokenize=False, add_generation_prompt=False
                )
                full_inputs = tokenizer(full_text, return_tensors="pt", padding=False).to(
                    model.device
                )
                full_len = full_inputs["input_ids"].shape[1]
                if full_len <= prompt_len:
                    n_skipped += 1
                    continue

                with torch.no_grad():
                    _ = model(**full_inputs)

                # Per-q row for B: (n_layers, D) fp16, mean-response
                if b_per_q_buf is not None:
                    row_layers = []
                    for lyr in layers:
                        hs = cap.captured[lyr]
                        resp = hs[0, prompt_len:full_len, :].float().cpu()
                        row_layers.append(resp.mean(dim=0))
                    b_per_q_buf.append(torch.stack(row_layers).to(torch.float16))

                # Per-q row for B*: (n_layers, D) fp16, mean-response excl. last token
                if bstar_per_q_buf is not None:
                    row_layers = []
                    for lyr in layers:
                        hs = cap.captured[lyr]
                        resp = hs[0, prompt_len:full_len, :].float().cpu()
                        if resp.shape[0] > 1:
                            row_layers.append(resp[:-1].mean(dim=0))
                        else:
                            row_layers.append(resp.mean(dim=0))
                    bstar_per_q_buf.append(torch.stack(row_layers).to(torch.float16))

                # Per-q row for R_per_token: (n_layers, n_response_positions, D) fp16
                if r_per_q_buf is not None:
                    layer_rows = []
                    for lyr in layers:
                        hs = cap.captured[lyr]
                        pos_vecs: list[torch.Tensor] = []
                        for t in response_positions:
                            tok_pos = prompt_len + t
                            if tok_pos >= full_len:
                                # Response shorter than this position; mask with NaN.
                                pos_vecs.append(torch.full((hs.shape[-1],), float("nan")))
                            else:
                                pos_vecs.append(hs[0, tok_pos, :].float().cpu())
                        layer_rows.append(torch.stack(pos_vecs))  # (n_pos, D)
                    r_per_q_buf.append(
                        torch.stack(layer_rows).to(torch.float16)
                    )  # (n_layers, n_pos, D)

                for lyr in layers:
                    hs = cap.captured[lyr]
                    resp_block = hs[0, prompt_len:full_len, :].float().cpu()
                    b_vec = resp_block.mean(dim=0)
                    bstar_vec = resp_block[:-1].mean(dim=0) if resp_block.shape[0] > 1 else b_vec
                    if b_accum is not None:
                        b_accum[lyr].append(b_vec)
                    if bstar_accum is not None:
                        bstar_accum[lyr].append(bstar_vec)
                    if b_train_accum is not None and is_train_q:
                        b_train_accum[lyr].append(b_vec)
                    if bstar_train_accum is not None and is_train_q:
                        bstar_train_accum[lyr].append(bstar_vec)
                    if r_accum is not None:
                        for t in response_positions:
                            tok_pos = prompt_len + t
                            if tok_pos >= full_len:
                                continue
                            r_vec = hs[0, tok_pos, :].float().cpu()
                            r_accum[(lyr, t)].append(r_vec)
                            if r_train_accum is not None and is_train_q:
                                r_train_accum[(lyr, t)].append(r_vec)

            if b_accum is not None:
                role_b: dict[tuple[int, int], torch.Tensor] = {}
                for lyr, vecs in b_accum.items():
                    if not vecs:
                        continue
                    centroid = torch.stack(vecs).mean(dim=0)
                    role_b[(lyr, 0)] = centroid
                    torch.save(
                        centroid,
                        output_dir / f"method_b__pos_0__layer_{lyr}" / f"{role_name}.pt",
                    )
                out["b"][role_name] = role_b
                if b_per_q_buf is not None and b_per_q_buf:
                    torch.save(
                        torch.stack(b_per_q_buf),
                        output_dir / "method_b" / f"{role_name}__per_q.pt",
                    )
                # Train-only centroid: (n_layers, D)
                if b_train_accum is not None:
                    n_layers = len(layers)
                    if b_per_q_buf is not None and b_per_q_buf:
                        D = b_per_q_buf[0].shape[-1]
                    else:
                        D = next(iter(role_b.values())).shape[-1] if role_b else 0
                    if D:
                        train_block = torch.full((n_layers, D), float("nan"))
                        for li, lyr in enumerate(layers):
                            vecs = b_train_accum[lyr]
                            if vecs:
                                train_block[li, :] = torch.stack(vecs).mean(dim=0)
                        torch.save(
                            train_block.float(),
                            output_dir / "method_b" / f"{role_name}__centroid_train.pt",
                        )

            if bstar_accum is not None:
                role_bstar: dict[tuple[int, int], torch.Tensor] = {}
                for lyr, vecs in bstar_accum.items():
                    if not vecs:
                        continue
                    centroid = torch.stack(vecs).mean(dim=0)
                    role_bstar[(lyr, 0)] = centroid
                    torch.save(
                        centroid,
                        output_dir / f"method_bstar__pos_0__layer_{lyr}" / f"{role_name}.pt",
                    )
                out["bstar"][role_name] = role_bstar
                if bstar_per_q_buf is not None and bstar_per_q_buf:
                    torch.save(
                        torch.stack(bstar_per_q_buf),
                        output_dir / "method_bstar" / f"{role_name}__per_q.pt",
                    )
                if bstar_train_accum is not None:
                    n_layers = len(layers)
                    if bstar_per_q_buf is not None and bstar_per_q_buf:
                        D = bstar_per_q_buf[0].shape[-1]
                    else:
                        D = next(iter(role_bstar.values())).shape[-1] if role_bstar else 0
                    if D:
                        train_block = torch.full((n_layers, D), float("nan"))
                        for li, lyr in enumerate(layers):
                            vecs = bstar_train_accum[lyr]
                            if vecs:
                                train_block[li, :] = torch.stack(vecs).mean(dim=0)
                        torch.save(
                            train_block.float(),
                            output_dir / "method_bstar" / f"{role_name}__centroid_train.pt",
                        )

            if r_accum is not None:
                role_r: dict[tuple[int, int], torch.Tensor] = {}
                for (lyr, t), vecs in r_accum.items():
                    if not vecs:
                        continue
                    centroid = torch.stack(vecs).mean(dim=0)
                    role_r[(lyr, t)] = centroid
                    torch.save(
                        centroid,
                        output_dir
                        / f"method_r_per_token__pos_{t}__layer_{lyr}"
                        / f"{role_name}.pt",
                    )
                out["r_per_token"][role_name] = role_r
                if r_per_q_buf is not None and r_per_q_buf:
                    torch.save(
                        torch.stack(r_per_q_buf),
                        output_dir / "method_r_per_token" / f"{role_name}__per_q.pt",
                    )
                if r_train_accum is not None:
                    n_layers = len(layers)
                    n_pos = len(response_positions)
                    if r_per_q_buf is not None and r_per_q_buf:
                        D = r_per_q_buf[0].shape[-1]
                    else:
                        D = next(iter(role_r.values())).shape[-1] if role_r else 0
                    if D:
                        train_block = torch.full((n_layers, n_pos, D), float("nan"))
                        for li, lyr in enumerate(layers):
                            for ti, t in enumerate(response_positions):
                                vecs = r_train_accum[(lyr, t)]
                                if vecs:
                                    train_block[li, ti, :] = torch.stack(vecs).mean(dim=0)
                        torch.save(
                            train_block.float(),
                            output_dir / "method_r_per_token" / f"{role_name}__centroid_train.pt",
                        )

            elapsed = time.time() - t0
            rate = (role_idx + 1) / elapsed * 60 if elapsed > 0 else 0.0
            print(
                f"  [{role_idx + 1}/{len(sorted_roles)}] {role_name} — "
                f"{elapsed:.0f}s elapsed, {rate:.1f} roles/min, {n_skipped} skipped",
                flush=True,
            )

    return out


# ── Method C variants (descriptive baselines from #201) ──────────────────────


def extract_method_c_variants(  # noqa: C901
    model,
    tokenizer,
    role_prompts: dict[str, list[str]],
    questions: list[str],
    layers: list[int],
    output_dir: Path,
    do_c1: bool,
    do_c2: bool,
    do_c3: bool,
    n_prompts: int = 1,
    save_per_q: bool = True,
    train_qid_set: set[int] | None = None,
) -> dict[str, dict[str, dict[tuple[int, int], torch.Tensor]]]:
    """Method C variants:
    - C1: hidden state at the LAST persona-prompt token (system prompt only, no question).
    - C2: hidden state at the LAST role-name token (e.g. 'assistant' literal).
    - C3: hidden state at the LAST tokenizer-output-token of role-name as a STANDALONE.

    These are descriptive-only baselines from #201; they share the prompt-side
    `add_generation_prompt=True` chat-template tail so we extract at i=-1 only.

    Per-question caches (round 3 / N2 fix):
        method_c1/<role>__per_q.pt   NOT WRITTEN — C1 has no question dep so the
                                     broadcast tile carried zero info. The analyzer's
                                     `load_per_q_at_cell` synthesizes from the
                                     cell-level `method_c1__pos_0__layer_<l>/<role>.pt`
                                     files on-demand (math is identical). Saves ~13.5 GB.
        method_c2/<role>__per_q.pt   NOT WRITTEN — same rationale as C1.
        method_c3/<role>__per_q.pt   shape (n_q, n_layers, D) fp16  — actual per-q
                                     (C3 IS question-dependent: stem template embeds
                                     the question, so the per-q tensor matters).

    Train-only centroids (round 2 / B2 fix; preserved in round 3):
        method_<m>/<role>__centroid_train.pt   shape (n_layers, D) fp32

    Output: out[c1|c2|c3][role][(layer, 0)] = (D,) tensor.
    """
    methods_run = [name for name, do in [("c1", do_c1), ("c2", do_c2), ("c3", do_c3)] if do]
    if not methods_run:
        return {"c1": {}, "c2": {}, "c3": {}}

    print(f"\n{'=' * 60}")
    print(f"Methods {methods_run}: C-family baselines")
    print(f"  Roles: {len(role_prompts)}, Questions: {len(questions)}, Layers: {len(layers)}")
    print(f"{'=' * 60}\n")

    for m in methods_run:
        for lyr in layers:
            (output_dir / f"method_{m}__pos_0__layer_{lyr}").mkdir(parents=True, exist_ok=True)
        if save_per_q:
            (output_dir / f"method_{m}").mkdir(parents=True, exist_ok=True)

    out: dict[str, dict[str, dict[tuple[int, int], torch.Tensor]]] = {
        "c1": {},
        "c2": {},
        "c3": {},
    }
    sorted_roles = sorted(role_prompts.items())
    t0 = time.time()

    with HiddenStateCapture(model, layers) as cap:
        for role_idx, (role_name, prompts) in enumerate(sorted_roles):
            sys_prompt = prompts[0] if prompts else ""

            # Round 3 / N2 fix: n_q dropped from this scope — only used in the
            # C1/C2 broadcast tiles, which we no longer write.
            n_layers = len(layers)

            # ── C1: system prompt only (no user question) — single forward pass ──
            if do_c1:
                # Build "system only" chat: just the system message + add_generation_prompt
                messages_c1 = [{"role": "system", "content": sys_prompt}]
                text_c1 = tokenizer.apply_chat_template(
                    messages_c1, tokenize=False, add_generation_prompt=True
                )
                inputs_c1 = tokenizer(text_c1, return_tensors="pt", padding=False).to(model.device)
                with torch.no_grad():
                    _ = model(**inputs_c1)
                last = inputs_c1["input_ids"].shape[1] - 1
                role_c1: dict[tuple[int, int], torch.Tensor] = {}
                # (n_layers, D) — same vector across all questions (no question dep)
                c1_layer_vecs: list[torch.Tensor] = []
                for lyr in layers:
                    vec = cap.captured[lyr][0, last, :].float().cpu()
                    role_c1[(lyr, 0)] = vec
                    c1_layer_vecs.append(vec)
                    torch.save(
                        vec,
                        output_dir / f"method_c1__pos_0__layer_{lyr}" / f"{role_name}.pt",
                    )
                out["c1"][role_name] = role_c1
                # Round 3 / N2 fix: do NOT write the (n_q, n_layers, D) broadcast tile —
                # C1 has no question dep, so the tile carried zero information beyond a
                # single (n_layers, D) vector. The analyzer's `load_per_q_at_cell`
                # synthesizes the per-q footprint on-demand from the cell-level files
                # in `method_c1__pos_0__layer_<l>/<role>.pt`. Saves ~13.5 GB at full
                # sweep size (275 roles x 240 q x 28 layers x D x 2 bytes).
                if save_per_q and c1_layer_vecs and train_qid_set is not None:
                    # Train-only centroid is still small + load-bearing for B2 fix.
                    train_block = torch.stack(c1_layer_vecs).float()  # (n_layers, D)
                    torch.save(
                        train_block,
                        output_dir / "method_c1" / f"{role_name}__centroid_train.pt",
                    )

            # ── C2: role-name as a STANDALONE token sequence (no chat template) ──
            if do_c2:
                inputs_c2 = tokenizer(role_name, return_tensors="pt", padding=False).to(
                    model.device
                )
                with torch.no_grad():
                    _ = model(**inputs_c2)
                last = inputs_c2["input_ids"].shape[1] - 1
                role_c2: dict[tuple[int, int], torch.Tensor] = {}
                c2_layer_vecs: list[torch.Tensor] = []
                for lyr in layers:
                    vec = cap.captured[lyr][0, last, :].float().cpu()
                    role_c2[(lyr, 0)] = vec
                    c2_layer_vecs.append(vec)
                    torch.save(
                        vec,
                        output_dir / f"method_c2__pos_0__layer_{lyr}" / f"{role_name}.pt",
                    )
                out["c2"][role_name] = role_c2
                # Round 3 / N2 fix: same rationale as C1 — no question dep, drop tile,
                # synthesize on-demand. Saves ~13.5 GB.
                if save_per_q and c2_layer_vecs and train_qid_set is not None:
                    train_block = torch.stack(c2_layer_vecs).float()
                    torch.save(
                        train_block,
                        output_dir / "method_c2" / f"{role_name}__centroid_train.pt",
                    )

            # ── C3: role-name in a sentence stem (averaged across questions for stability) ──
            if do_c3:
                accum_c3: dict[int, list[torch.Tensor]] = {lyr: [] for lyr in layers}
                # Per-q buffer for C3: (n_q, n_layers, D) fp16
                c3_per_q_buf: list[torch.Tensor] | None = [] if save_per_q else None
                # Train-only accumulator (B2 fix)
                c3_train_accum: dict[int, list[torch.Tensor]] | None = (
                    {lyr: [] for lyr in layers} if (train_qid_set is not None) else None
                )
                stem_template = "The following is a description of a {role}.\n\n{question}"
                for q_idx, question in enumerate(questions):
                    text_c3 = stem_template.format(role=role_name, question=question)
                    inputs_c3 = tokenizer(text_c3, return_tensors="pt", padding=False).to(
                        model.device
                    )
                    with torch.no_grad():
                        _ = model(**inputs_c3)
                    last = inputs_c3["input_ids"].shape[1] - 1
                    is_train_q = train_qid_set is not None and q_idx in train_qid_set
                    layer_row: list[torch.Tensor] = []
                    for lyr in layers:
                        vec = cap.captured[lyr][0, last, :].float().cpu()
                        accum_c3[lyr].append(vec)
                        layer_row.append(vec)
                        if c3_train_accum is not None and is_train_q:
                            c3_train_accum[lyr].append(vec)
                    if c3_per_q_buf is not None:
                        c3_per_q_buf.append(torch.stack(layer_row).to(torch.float16))
                role_c3: dict[tuple[int, int], torch.Tensor] = {}
                for lyr, vecs in accum_c3.items():
                    if not vecs:
                        continue
                    centroid = torch.stack(vecs).mean(dim=0)
                    role_c3[(lyr, 0)] = centroid
                    torch.save(
                        centroid,
                        output_dir / f"method_c3__pos_0__layer_{lyr}" / f"{role_name}.pt",
                    )
                out["c3"][role_name] = role_c3
                if c3_per_q_buf is not None and c3_per_q_buf:
                    torch.save(
                        torch.stack(c3_per_q_buf),
                        output_dir / "method_c3" / f"{role_name}__per_q.pt",
                    )
                if c3_train_accum is not None and role_c3:
                    D = next(iter(role_c3.values())).shape[-1]
                    train_block = torch.full((n_layers, D), float("nan"))
                    for li, lyr in enumerate(layers):
                        vecs = c3_train_accum[lyr]
                        if vecs:
                            train_block[li, :] = torch.stack(vecs).mean(dim=0)
                    torch.save(
                        train_block.float(),
                        output_dir / "method_c3" / f"{role_name}__centroid_train.pt",
                    )

            elapsed = time.time() - t0
            rate = (role_idx + 1) / elapsed * 60 if elapsed > 0 else 0.0
            print(
                f"  [{role_idx + 1}/{len(sorted_roles)}] {role_name} — "
                f"{elapsed:.0f}s elapsed, {rate:.1f} roles/min",
                flush=True,
            )

    return out


# ── CAA wrapper (descriptive only per plan §3 v3 fix 1) ──────────────────────


def extract_caa_in_process(
    model,
    tokenizer,
    role_prompts: dict[str, list[str]],
    questions: list[str],
    layers: list[int],
    prompt_positions: list[int],
    output_dir: Path,
    n_prompts: int,
) -> None:
    """CAA via the in-process implementation (delegates to run_caa.py's helpers).

    Imported lazily so the rest of the sweep does not depend on run_caa import-time
    side effects.
    """
    # Lazy import to keep run_caa optional at import time
    from run_caa import extract_caa as _extract_caa

    _extract_caa(
        model,
        tokenizer,
        role_prompts,
        questions,
        layers,
        prompt_positions,
        output_dir,
        n_prompts=n_prompts,
    )


# ── Argparse helpers ─────────────────────────────────────────────────────────


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_method_list(s: str) -> list[str]:
    parts = [m.strip().lower() for m in s.split(",") if m.strip()]
    valid = {"a", "a_per_token", "b", "bstar", "c1", "c2", "c3", "r_per_token", "caa"}
    bad = [m for m in parts if m not in valid]
    if bad:
        raise ValueError(f"Unknown methods: {bad}. Valid: {sorted(valid)}")
    return parts


# ── Main ─────────────────────────────────────────────────────────────────────


def main():  # noqa: C901
    parser = argparse.ArgumentParser(
        description="Continuous (method x token x layer) sweep of persona-vector extraction (#263)"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model name or path")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for HF forward passes (cpu = smoke / dev only)",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU index when --device cuda")
    parser.add_argument(
        "--layers",
        type=str,
        default=",".join(str(x) for x in DEFAULT_LAYERS_FULL),
        help="Comma-separated layer indices",
    )
    parser.add_argument(
        "--prompt-token-positions",
        type=str,
        default=",".join(str(x) for x in DEFAULT_PROMPT_POSITIONS),
        help="Comma-separated prompt-side token offsets, e.g. '-5,-4,-3,-2,-1'",
    )
    parser.add_argument(
        "--response-token-positions",
        type=str,
        default=",".join(str(x) for x in DEFAULT_RESPONSE_POSITIONS),
        help="Comma-separated response-side token indices, e.g. '0,1,2,4,8,16,32,64,128'",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated method names. "
        "Subset of: a, a_per_token, b, bstar, c1, c2, c3, r_per_token, caa",
    )
    parser.add_argument("--n-prompts", type=int, default=1, help="System prompts per role")
    parser.add_argument("--n-questions", type=int, default=None, help="Questions (default: 240)")
    parser.add_argument(
        "--n-personas",
        type=int,
        default=None,
        help="Subset to first N sorted personas (smoke / dev)",
    )
    parser.add_argument(
        "--roles",
        type=str,
        default=None,
        help="Comma-separated explicit roles (overrides --n-personas)",
    )
    parser.add_argument(
        "--reuse-cache",
        type=str,
        default=None,
        help="Path to #218 cache root for cache-shape assertion + Method B reuse",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max generated tokens for Method B / R_per_token (greedy, T=0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Centroid + per-q-cache root directory (e.g. data/persona_vectors/issue_263/...)",
    )
    parser.add_argument(
        "--train-qids",
        type=str,
        default="0..199",
        help="Train question slice (used for B2 fix: train-only centroid output). "
        "'0..199' gives indices 0..199 inclusive. Set to empty string to disable train-only.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: 2 personas, 4 questions, 2 layers, 2 prompt + 2 response positions",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.smoke:
        # Override key args for smoke
        args.layers = "0,7"
        args.prompt_token_positions = "-1,-2"
        args.response_token_positions = "0,1"
        args.n_questions = 4
        args.n_personas = 2
        args.max_new_tokens = 8
        args.train_qids = "0..1"  # 2 train, 1 val, 1 test in smoke
        # If user did not specify methods, default smoke to a,caa (the cheapest)
        if args.methods == ",".join(DEFAULT_METHODS):
            args.methods = "a,caa"
        print(
            f"SMOKE MODE: layers={args.layers}, "
            f"prompt_pos={args.prompt_token_positions}, "
            f"response_pos={args.response_token_positions}, "
            f"n_q=4, n_personas=2, methods={args.methods}"
        )

    layers = parse_int_list(args.layers)
    prompt_positions = parse_int_list(args.prompt_token_positions)
    response_positions = parse_int_list(args.response_token_positions)
    methods = parse_method_list(args.methods)

    # Train-qid slice for the B2 fix (train-only centroid emission). Empty string disables.
    train_qid_set: set[int] | None = None
    if args.train_qids:
        s = args.train_qids.strip()
        if ".." in s:
            lo, hi = s.split("..")
            train_qid_set = set(range(int(lo), int(hi) + 1))
        else:
            train_qid_set = {int(x.strip()) for x in s.split(",") if x.strip()}

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 0b cache-shape assertion (if --reuse-cache provided) ──
    if args.reuse_cache:
        reuse_root = Path(args.reuse_cache)
        # Defensive: only assert against method_a per_q caches; the plan calls for that one.
        n_q_for_assert = args.n_questions or 240
        assert_cache_shape(
            reuse_root,
            method="a",
            expected_n_q=n_q_for_assert,
            expected_n_layers=DEFAULT_N_LAYERS,
            expected_hidden_dim=DEFAULT_HIDDEN_DIM,
        )

    # ── Resolve roles filter ──
    roles_filter: list[str] | None = None
    if args.roles:
        roles_filter = [r.strip() for r in args.roles.split(",")]
    elif args.n_personas:
        with open(DATA_DIR / "role_list.json") as f:
            all_roles = sorted(json.load(f).keys())
        roles_filter = all_roles[: args.n_personas]

    print("Loading roles + questions...")
    role_prompts = load_roles(roles_filter)
    questions = load_extraction_questions(args.n_questions)
    print(f"  {len(role_prompts)} roles, {len(questions)} questions")

    # ── Load model ──
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

    metadata = _build_run_metadata(args, n_roles=len(role_prompts), n_questions=len(questions))
    with open(output_dir / "sweep_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Method A / A_per_token (single forward pass dumps all prompt positions) ──
    if "a" in methods or "a_per_token" in methods:
        # Treat 'a' and 'a_per_token' as the same forward-pass; positions argument
        # already governs which cells we dump. If only 'a' was requested (no
        # a_per_token), we still dump only the requested positions.
        extract_prompt_side_grid(
            model,
            tokenizer,
            role_prompts,
            questions,
            layers,
            prompt_positions,
            output_dir,
            n_prompts=args.n_prompts,
            save_per_q=True,
            train_qid_set=train_qid_set,
        )

    # ── Method B / B* / R_per_token (response-side) ──
    do_b = "b" in methods
    do_bstar = "bstar" in methods
    do_r = "r_per_token" in methods
    if do_b or do_bstar or do_r:
        # Need responses. Try cache reuse first iff Method B was requested.
        responses_path = output_dir / "method_b" / "generated_responses.json"
        if args.reuse_cache:
            reuse_responses = Path(args.reuse_cache) / "method_b" / "generated_responses.json"
            if reuse_responses.exists() and not responses_path.exists():
                # Surface the path so generate_responses_vllm picks it up
                responses_path.parent.mkdir(parents=True, exist_ok=True)
                # Symlink (cheap, no copy) — fall back to copy if symlink unsupported
                try:
                    responses_path.symlink_to(reuse_responses.resolve())
                    print(f"  Linked reuse-cache responses -> {responses_path}")
                except OSError:
                    import shutil

                    shutil.copy2(reuse_responses, responses_path)
                    print(f"  Copied reuse-cache responses -> {responses_path}")

        responses = generate_responses_vllm(
            args.model,
            role_prompts,
            questions,
            n_prompts=args.n_prompts,
            gpu_id=args.gpu_id,
            output_path=responses_path,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        extract_response_methods(
            model,
            tokenizer,
            responses,
            layers,
            response_positions,
            output_dir,
            do_b=do_b,
            do_bstar=do_bstar,
            do_r_per_token=do_r,
            save_per_q=True,
            train_qid_set=train_qid_set,
        )

    # ── Method C variants ──
    do_c1 = "c1" in methods
    do_c2 = "c2" in methods
    do_c3 = "c3" in methods
    if do_c1 or do_c2 or do_c3:
        extract_method_c_variants(
            model,
            tokenizer,
            role_prompts,
            questions,
            layers,
            output_dir,
            do_c1=do_c1,
            do_c2=do_c2,
            do_c3=do_c3,
            n_prompts=args.n_prompts,
            save_per_q=True,
            train_qid_set=train_qid_set,
        )

    # ── CAA (descriptive only per plan §3 v3 fix 1) ──
    if "caa" in methods:
        # CAA is on the same prompt_positions x layers grid as Method A.
        extract_caa_in_process(
            model,
            tokenizer,
            role_prompts,
            questions,
            layers,
            prompt_positions,
            output_dir,
            n_prompts=args.n_prompts,
        )

    # ── Done ──
    cells_manifest = build_cells_manifest(
        output_dir, methods, layers, prompt_positions, response_positions
    )
    with open(output_dir / "cells_manifest.json", "w") as f:
        json.dump(cells_manifest, f, indent=2)
    print(f"\nSweep done. Cells manifest: {output_dir}/cells_manifest.json")
    print(f"  Total cells: {sum(cells_manifest['cells_per_method'].values())}")


def build_cells_manifest(
    output_dir: Path,
    methods: list[str],
    layers: list[int],
    prompt_positions: list[int],
    response_positions: list[int],
) -> dict:
    """Manifest of how many (method, position, layer) cells were actually written.

    Per plan §5 / §7: H1's denominator (pre-registered = 672) must agree with this
    manifest within 1 cell, otherwise H1 reporting aborts in the analyzer.
    """
    cells_per_method: dict[str, int] = {}
    cell_paths: dict[str, list[str]] = {}
    for m in methods:
        if m == "a" or m == "a_per_token":
            poses = prompt_positions
            method_dir_pattern = "method_a"
        elif m == "caa":
            poses = prompt_positions
            method_dir_pattern = "method_caa"
        elif m == "r_per_token":
            poses = response_positions
            method_dir_pattern = "method_r_per_token"
        else:  # b, bstar, c1, c2, c3 -> single position slot 0
            poses = [0]
            method_dir_pattern = f"method_{m}"
        count = 0
        paths: list[str] = []
        for p in poses:
            for lyr in layers:
                cell_dir = output_dir / f"{method_dir_pattern}__pos_{p}__layer_{lyr}"
                if cell_dir.exists():
                    n = len(list(cell_dir.glob("*.pt")))
                    if n > 0:
                        count += 1
                        paths.append(str(cell_dir))
        cells_per_method[m] = count
        cell_paths[m] = paths

    return {
        "cells_per_method": cells_per_method,
        "cell_paths": cell_paths,
        "n_layers": len(layers),
        "n_prompt_positions": len(prompt_positions),
        "n_response_positions": len(response_positions),
    }


if __name__ == "__main__":
    main()
