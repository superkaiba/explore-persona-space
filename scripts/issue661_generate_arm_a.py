#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ĉ, ρ, →, ×) in scientific docstrings + log messages.
"""Issue #661 P1 (POD GPU): arm-A on-policy generation under pos/neg instruction.

For each behavior: 48 extraction probes × {5 pos instructions, 5 neg
instructions} × 10 rollouts, temp 1.0, max_new_tokens=512, via ONE batched vLLM
``LLM.generate`` call with ``SamplingParams(n=10)`` (CLAUDE.md: never sequential
HF generate). Persona / instruction injection is ALWAYS a system turn.

The output is the raw generation store the P2 judge-filter consumes. One JSON per
(behavior, polarity) under ``eval_results/issue_661/raw_completions/`` — each
records, per (instruction_idx, probe), the 10 rollout completions. This is the
canonical raw-completions artifact (uploaded to HF before pod terminate).

Reuses the #658 ``vllm_generate`` / ``_reap_vllm`` teardown (the vLLM→HF
co-residence OOM guard) — one engine per phase, reaped before the P3 HF load.

Usage::

    uv run python scripts/issue661_generate_arm_a.py \
        --behaviors sycophancy refusal broad_em --gpu-id 0

    # local CPU smoke (tiny model, HF generate, 1 probe × 1 rollout):
    uv run python scripts/issue661_generate_arm_a.py --behaviors sycophancy \
        --model Qwen/Qwen2.5-0.5B-Instruct --device cpu --no-vllm \
        --n-probes 1 --n-instruction-pairs 1 --n-rollouts 1 --no-upload \
        --instructions-dir /tmp/i661_smoke --out-dir /tmp/i661_smoke
"""

from __future__ import annotations

import os

# VLLM_WORKER_MULTIPROC_METHOD=spawn BEFORE any `import vllm` — the dispatcher
# loads transformers/tokenizer before LLM(), which poisons a fork()ed
# EngineCore (gotchas.md #628). Set at module top, before the lazy vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402

# Cross-script helpers hoisted to module top (gotchas.md #606): the #658
# vLLM teardown reused here.
from issue658_extract_base_store import _reap_vllm  # noqa: E402
from issue661_common import (  # noqa: E402
    DATA_DIR,
    DEFAULT_MODEL,
    EVAL_RESULTS_DIR,
    HF_DATA_REPO,
    HF_PREFIX,
    MAX_NEW_TOKENS,
    N_ROLLOUTS,
    ROLLOUT_TEMPERATURE,
    dump_json,
    instructions_path,
    load_json,
    probe_pool_path,
    system_prompt_messages,
)

load_dotenv(str(PROJECT_ROOT / ".env"))
logger = logging.getLogger("issue661_gen_a")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def raw_completions_path(out_root: Path, behavior: str, polarity: str) -> Path:
    return out_root / "raw_completions" / f"{behavior}__{polarity}.json"


def build_cells(
    tokenizer,
    instructions: list[dict],
    probes: list[str],
    polarity: str,
) -> tuple[list[str], list[tuple[int, int]]]:
    """Templated prompt strings for (instruction_idx, probe_idx) + the index.

    polarity in {"pos", "neg"} selects which instruction text of each pair is
    used as the SYSTEM prompt. Returns (prompt_texts, index) where
    index[i] = (instruction_idx, probe_idx).
    """
    prompts: list[str] = []
    index: list[tuple[int, int]] = []
    for ii, pair in enumerate(instructions):
        system_prompt = pair[polarity]
        for pi, q in enumerate(probes):
            messages = system_prompt_messages(system_prompt, q)
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)
            index.append((ii, pi))
    return prompts, index


def vllm_generate_n(
    model_name: str, prompts: list[str], n: int, temperature: float, max_tokens: int
):
    """vLLM batched generation with n samples per prompt in ONE call.

    Returns the list of RequestOutputs (one per prompt, each with n outputs).
    use_tqdm=False (gotchas.md #613 ZeroDivisionError). Engine reaped by caller
    of the calling phase (one engine per phase) — here we reap immediately
    because P1 owns its own engine lifetime.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name, dtype="bfloat16", gpu_memory_utilization=0.45)
    sp = SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens)
    try:
        outs = llm.generate(prompts, sp, use_tqdm=False)
        # Materialize the texts before reaping the engine.
        result = [[o.text for o in req.outputs] for req in outs]
    finally:
        _reap_vllm(llm)
    return result


def hf_generate_n(
    model, tokenizer, prompts: list[str], n: int, temperature: float, max_tokens: int
):
    """HF generate fallback (CPU smoke / --no-vllm). n samples per prompt."""
    results: list[list[str]] = []
    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)
        samples: list[str] = []
        for _ in range(n):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=(temperature if temperature > 0 else None),
                    top_p=(0.95 if temperature > 0 else None),
                )
            gen = out[0, inputs["input_ids"].shape[1] :]
            samples.append(tokenizer.decode(gen, skip_special_tokens=True))
        results.append(samples)
    return results


def generate_behavior(
    behavior: str,
    *,
    instructions_dir: Path,
    out_root: Path,
    model_name: str,
    use_vllm: bool,
    n_probes: int,
    n_instruction_pairs: int,
    n_rollouts: int,
    max_tokens: int,
    hf_model=None,
    hf_tokenizer=None,
) -> dict:
    """Generate arm-A completions for one behavior (both polarities)."""
    instr = load_json(
        instructions_dir / f"instructions_{behavior}.json"
        if instructions_dir
        else instructions_path(behavior)
    )
    pool = load_json(
        instructions_dir / f"probe_pool_{behavior}.json"
        if instructions_dir
        else probe_pool_path(behavior)
    )
    instructions = (
        instr["instruction"][:n_instruction_pairs] if n_instruction_pairs else instr["instruction"]
    )
    probes = pool["extraction_questions"][:n_probes] if n_probes else pool["extraction_questions"]

    # Build a tokenizer for templating (the vLLM path also needs it).
    if hf_tokenizer is not None:
        tokenizer = hf_tokenizer
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)

    summary: dict = {"behavior": behavior, "polarities": {}}
    for polarity in ("pos", "neg"):
        out_path = raw_completions_path(out_root, behavior, polarity)
        if out_path.exists():
            logger.info("%s/%s: already generated at %s — skip", behavior, polarity, out_path)
            existing = load_json(out_path)
            summary["polarities"][polarity] = {
                "n_cells": len(existing["cells"]),
                "skipped": True,
            }
            continue
        prompts, index = build_cells(tokenizer, instructions, probes, polarity)
        t0 = time.time()
        if use_vllm:
            per_prompt = vllm_generate_n(
                model_name, prompts, n_rollouts, ROLLOUT_TEMPERATURE, max_tokens
            )
        else:
            per_prompt = hf_generate_n(
                hf_model, tokenizer, prompts, n_rollouts, ROLLOUT_TEMPERATURE, max_tokens
            )
        cells = []
        for (ii, pi), samples in zip(index, per_prompt, strict=True):
            cells.append(
                {
                    "instruction_idx": ii,
                    "probe_idx": pi,
                    "probe": probes[pi],
                    "rollouts": [{"text": s} for s in samples],
                }
            )
        dump_json(
            {
                "behavior": behavior,
                "polarity": polarity,
                "arm": "A_onpolicy_instr",
                "n_instruction_pairs": len(instructions),
                "n_probes": len(probes),
                "n_rollouts": n_rollouts,
                "temperature": ROLLOUT_TEMPERATURE,
                "max_new_tokens": max_tokens,
                "probe_pool_sha": pool.get("sha256"),
                "instructions_sha": instr.get("sha256"),
                "cells": cells,
                "metadata": reproducibility_metadata({"script": "issue661_generate_arm_a"}),
            },
            out_path,
        )
        logger.info(
            "%s/%s: %d cells × %d rollouts → %s (%.1fs)",
            behavior,
            polarity,
            len(cells),
            n_rollouts,
            out_path,
            time.time() - t0,
        )
        summary["polarities"][polarity] = {"n_cells": len(cells), "skipped": False}
    return summary


def upload_raw_completions(out_root: Path) -> None:
    """Bulk-upload the arm-A raw completions to the HF data repo (one commit)."""
    from huggingface_hub import HfApi

    rc_dir = out_root / "raw_completions"
    if not rc_dir.is_dir():
        logger.warning("no raw_completions dir at %s — nothing to upload", rc_dir)
        return
    api = HfApi()
    api.upload_folder(
        folder_path=str(rc_dir),
        path_in_repo=f"{HF_PREFIX}/raw_completions",
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        commit_message="issue661: arm-A raw completions (P1)",
    )
    files = [
        f
        for f in api.list_repo_files(HF_DATA_REPO, repo_type="dataset")
        if f.startswith(f"{HF_PREFIX}/raw_completions/")
    ]
    n_local = len(list(rc_dir.glob("*.json")))
    if len(files) < n_local:
        raise RuntimeError(
            f"raw-completions upload verification failed: remote has {len(files)} files under "
            f"{HF_PREFIX}/raw_completions/, local has {n_local} (.json)"
        )
    logger.info("uploaded + verified %d raw-completion files to HF", len(files))


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #661 P1: arm-A on-policy generation.")
    ap.add_argument("--behaviors", nargs="+", default=["sycophancy", "refusal", "broad_em"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--no-vllm", action="store_true", help="HF generate fallback (CPU smoke)")
    ap.add_argument("--n-probes", type=int, default=0, help="cap probes (0 = full pool)")
    ap.add_argument("--n-instruction-pairs", type=int, default=0, help="cap pairs (0 = all 5)")
    ap.add_argument("--n-rollouts", type=int, default=N_ROLLOUTS)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument(
        "--instructions-dir",
        type=Path,
        default=None,
        help="dir holding instructions_<b>.json / probe_pool_<b>.json (default DATA_DIR)",
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="override eval_results dir (smoke)")
    ap.add_argument("--no-upload", action="store_true", help="skip HF upload (smoke)")
    args = ap.parse_args()

    out_root = args.out_dir or EVAL_RESULTS_DIR
    instructions_dir = args.instructions_dir or DATA_DIR

    if args.device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    use_vllm = not args.no_vllm and use_cuda

    hf_model = hf_tokenizer = None
    if not use_vllm:
        # CPU smoke / --no-vllm: load HF model once for the generate fallback.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(args.model)
        dtype = torch.bfloat16 if use_cuda else torch.float32
        device_map = {"": torch.device("cuda:0")} if use_cuda else None
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, device_map=device_map
        )
        hf_model.eval()

    for behavior in args.behaviors:
        generate_behavior(
            behavior,
            instructions_dir=instructions_dir,
            out_root=out_root,
            model_name=args.model,
            use_vllm=use_vllm,
            n_probes=args.n_probes,
            n_instruction_pairs=args.n_instruction_pairs,
            n_rollouts=args.n_rollouts,
            max_tokens=args.max_new_tokens,
            hf_model=hf_model,
            hf_tokenizer=hf_tokenizer,
        )

    if not args.no_upload:
        upload_raw_completions(out_root)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
