#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
"""Issue #490 PHASE 1 — on-policy R generation (cached, resumable).

Per plan v1 §4.5 PHASE 1:

  * For each of the 55 unique personas in POOL_16 ∪ NEGATIVES_FIXED ∪
    HELD_OUT_35 (BIT-IDENTICAL to #478), generate greedy (temp=0) responses
    on every question in DATA_QUESTIONS ∪ EVAL_QUESTIONS (40 train + 20
    eval = 60 q/persona).
  * **Re-uses #478's on-policy R cache** at
    ``data/issue_478/onpolicy_R/{persona}.json`` if present (no-op when the
    cache is linked from a worktree).
  * Writes new caches at ``data/issue_490/onpolicy_R/{persona}.json``
    AND symlinks/copies #478's existing files to that path for downstream
    Phase 2 (so the training-data builder always reads from a single
    canonical dir).
  * Per-(persona, q) R-cap: max_new_tokens = MAX_LENGTH − prompt_len −
    R_CAP_SAFETY_MARGIN; FAIL LOUD if any per-q cap < R_CAP_MIN.

CLI mirrors #478's generator.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

log = bootstrap()

from _issue490_common import (  # noqa: E402
    ALL_PERSONAS,
    BASE_MODEL,
    MAX_LENGTH,
    R_CAP_MIN,
    R_CAP_SAFETY_MARGIN,
    assert_marker_token_id,
    load_all_persona_prompts,
)


def _import_questions() -> tuple[list[str], list[str]]:
    """Re-use #405/#478's question banks for cross-experiment comparability."""
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from run_leakage_v3_onpolicy import DATA_QUESTIONS, EVAL_QUESTIONS

    return list(DATA_QUESTIONS), list(EVAL_QUESTIONS)


def compute_prompt_len(tokenizer, persona_prompt: str, question: str) -> int:
    messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return len(tokenizer.encode(prompt_text, add_special_tokens=False))


def _link_478_cache(persona: str, out_dir: Path, src_dir_478: Path) -> bool:
    """If #478 already cached this persona's R, copy/link it to out_dir.

    Returns True if a #478 cache file was successfully reused, False if no
    #478 cache exists for this persona.
    """
    src = src_dir_478 / f"{persona}.json"
    if not src.exists():
        return False
    dst = out_dir / f"{persona}.json"
    if dst.exists():
        return True
    # Copy (not symlink — the data is small and copying keeps the #490
    # cache directory self-contained when archived).
    shutil.copy2(src, dst)
    return True


def main() -> int:  # noqa: C901 — argparse + per-prompt R-cap loop
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--personas", type=str, default="")
    parser.add_argument("--questions", type=str, default="both", choices=["train", "eval", "both"])
    parser.add_argument("--max-personas", type=int, default=0, help="0 = no cap")
    parser.add_argument("--max-questions", type=int, default=0, help="0 = no cap")
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.55")),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "issue_490" / "onpolicy_R"),
    )
    parser.add_argument(
        "--src-dir-478",
        type=str,
        default=str(PROJECT_ROOT / "data" / "issue_478" / "onpolicy_R"),
        help="Inherit #478's cache from this dir (default: data/issue_478/onpolicy_R).",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-vllm",
        action="store_true",
        help="Only link/copy from #478 cache; do NOT load vLLM (used when the "
        "linked cache covers every needed persona).",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir_478 = Path(args.src_dir_478)

    all_prompts = load_all_persona_prompts()
    if args.personas:
        personas = [p.strip() for p in args.personas.split(",") if p.strip()]
        missing = [p for p in personas if p not in all_prompts]
        if missing:
            raise SystemExit(f"Unknown personas: {missing!r}")
    else:
        personas = list(ALL_PERSONAS)
    if args.max_personas:
        personas = personas[: args.max_personas]

    train_qs, eval_qs = _import_questions()
    if args.questions == "train":
        questions = train_qs
    elif args.questions == "eval":
        questions = eval_qs
    else:
        questions = train_qs + eval_qs
    if args.max_questions:
        questions = questions[: args.max_questions]

    # ── Phase 1a: link/copy #478 cache files where available ──────────────
    n_linked = 0
    for p in personas:
        if _link_478_cache(p, out_dir, src_dir_478):
            n_linked += 1
    log.info(
        "Linked %d of %d personas from #478 cache at %s → %s",
        n_linked,
        len(personas),
        src_dir_478,
        out_dir,
    )

    # Skip personas already cached (either via link or earlier run) unless --force.
    to_run = []
    for p in personas:
        cached = out_dir / f"{p}.json"
        if cached.exists() and not args.force:
            continue
        to_run.append(p)
    if not to_run:
        log.info("All %d personas cached. Nothing to do.", len(personas))
        return 0

    if args.skip_vllm:
        raise SystemExit(
            f"--skip-vllm but {len(to_run)} personas still need generation: "
            f"{to_run!r}. Drop --skip-vllm and run with --gpu."
        )

    log.info(
        "Phase 1 R-gen: %d personas (uncached) × %d questions on GPU %d (cap=%d − prompt_len − %d)",
        len(to_run),
        len(questions),
        args.gpu,
        MAX_LENGTH,
        R_CAP_SAFETY_MARGIN,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    assert_marker_token_id(tokenizer)

    from vllm import LLM, SamplingParams

    log.info("Loading vLLM (gpu_mem_util=%.2f) ...", args.gpu_mem_util)
    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=MAX_LENGTH,
        max_num_seqs=64,
        seed=42,
    )

    prompts: list[str] = []
    keys: list[tuple[str, str]] = []
    r_caps: list[int] = []
    for persona in to_run:
        sys_prompt = all_prompts[persona]
        for q in questions:
            prompt_len = compute_prompt_len(tokenizer, sys_prompt, q)
            r_cap = MAX_LENGTH - prompt_len - R_CAP_SAFETY_MARGIN
            if r_cap < R_CAP_MIN:
                raise RuntimeError(
                    f"R-cap too small for persona={persona!r} q={q[:60]!r}: "
                    f"prompt_len={prompt_len}, r_cap={r_cap} < {R_CAP_MIN}"
                )
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)
            keys.append((persona, q))
            r_caps.append(r_cap)

    log.info("Built %d prompts; running vLLM batched greedy ...", len(prompts))
    sampling = [SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=cap) for cap in r_caps]
    outputs = llm.generate(prompts, sampling)

    per_persona: dict[str, dict[str, str]] = {p: {} for p in to_run}
    truncation_count = 0
    for out, (persona, q) in zip(outputs, keys, strict=True):
        if out.outputs[0].finish_reason == "length":
            truncation_count += 1
        per_persona[persona][q] = out.outputs[0].text
    log.info("Generation done. %d / %d prompts hit length cap.", truncation_count, len(prompts))

    for persona, qmap in per_persona.items():
        cached_path = out_dir / f"{persona}.json"
        cached_path.write_text(
            json.dumps(
                {
                    "persona": persona,
                    "n_questions": len(qmap),
                    "max_length": MAX_LENGTH,
                    "r_cap_safety_margin": R_CAP_SAFETY_MARGIN,
                    "responses": qmap,
                },
                indent=2,
            )
        )
        log.info("Wrote %s (%d responses)", cached_path.name, len(qmap))

    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        log.warning("torch.cuda.empty_cache() raised; continuing")
    log.info("Phase 1 done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
