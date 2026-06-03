#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
"""Issue #478 PHASE 1 — on-policy R generation (one-time, cached).

Per plan v5 §4.8 PHASE 1:

  * For each of the 55 unique personas in POOL_16 ∪ NEGATIVES_FIXED ∪ HELD_OUT_35,
    generate greedy (temp=0) responses on every question in
    ``DATA_QUESTIONS ∪ EVAL_QUESTIONS`` (40 train + 20 eval = 60 q/persona).
  * Fix D (inherited from #405): per-(persona, question) R-cap
    ``max_new_tokens = MAX_LENGTH − prompt_len − R_CAP_SAFETY_MARGIN``
    so the resulting ``prompt + R + " ※" + EOS`` fits inside training
    ``max_length=1024`` with margin. FAIL LOUD if any per-q R-cap < R_CAP_MIN.
  * vLLM batched on a single H100 (~75 min wall per plan §9; scales linearly
    from #405's 20 personas × 60 q in ~30 min).
  * Cached at ``data/issue_478/onpolicy_R/{persona}.json`` (resumable — skips
    personas already on disk; --force overrides).

CLI:
  --gpu N            GPU index for this process (sets CUDA_VISIBLE_DEVICES).
  --personas STR     Comma-separated persona names; defaults to all 55.
  --questions tag    "train" / "eval" / "both" (default: both).
  --max-personas N   Smoke flag: only generate for the first N personas.
  --max-questions N  Smoke flag: only generate for the first N questions.
  --gpu-mem-util F   vLLM GPU memory utilization (default: env VLLM_GPU_MEM_UTIL or 0.55).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

log = bootstrap()

from _issue478_common import (  # noqa: E402
    ALL_PERSONAS,
    BASE_MODEL,
    MAX_LENGTH,
    R_CAP_MIN,
    R_CAP_SAFETY_MARGIN,
    assert_marker_token_id,
    load_all_persona_prompts,
)


def _import_questions() -> tuple[list[str], list[str]]:
    """Reuse #405's question banks for cross-experiment comparability."""
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from run_leakage_v3_onpolicy import DATA_QUESTIONS, EVAL_QUESTIONS

    return list(DATA_QUESTIONS), list(EVAL_QUESTIONS)


def compute_prompt_len(tokenizer, persona_prompt: str, question: str) -> int:
    """Tokenize the chat-template-wrapped prompt + return token length."""
    messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return len(tokenizer.encode(prompt_text, add_special_tokens=False))


def main() -> int:  # noqa: C901 — argparse + per-prompt R-cap loop is sequential
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (CUDA_VISIBLE_DEVICES)")
    parser.add_argument(
        "--personas",
        type=str,
        default="",
        help="Comma-separated persona names (default: all 55)",
    )
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
        default=str(PROJECT_ROOT / "data" / "issue_478" / "onpolicy_R"),
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-generate even if cached file exists"
    )
    args = parser.parse_args()

    # Pin GPU BEFORE any torch import.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persona list
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

    # Question banks
    train_qs, eval_qs = _import_questions()
    if args.questions == "train":
        questions = train_qs
    elif args.questions == "eval":
        questions = eval_qs
    else:
        questions = train_qs + eval_qs
    if args.max_questions:
        questions = questions[: args.max_questions]

    log.info(
        "Phase 1 R-gen: %d personas × %d questions on GPU %d (cap=%d − prompt_len − %d)",
        len(personas),
        len(questions),
        args.gpu,
        MAX_LENGTH,
        R_CAP_SAFETY_MARGIN,
    )

    # ── Tokenizer + marker assert (sentinel pre-launch check) ─────────────
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    assert_marker_token_id(tokenizer)

    # Skip personas already cached unless --force.
    to_run = []
    for p in personas:
        cached = out_dir / f"{p}.json"
        if cached.exists() and not args.force:
            log.info("Cached, skipping: %s", cached.name)
            continue
        to_run.append(p)
    if not to_run:
        log.info("All personas already cached. Nothing to do.")
        return 0

    # ── vLLM load ────────────────────────────────────────────────────────
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

    # ── Build per-(persona, question) prompts + per-prompt R-cap ─────────
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
                    f"prompt_len={prompt_len}, r_cap={r_cap} < {R_CAP_MIN}. "
                    f"Either raise MAX_LENGTH or shorten the persona prompt."
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

    log.info("Built %d (persona, question) prompts; running vLLM batched greedy ...", len(prompts))

    sampling = [SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=cap) for cap in r_caps]
    outputs = llm.generate(prompts, sampling)

    # ── Re-key by (persona, question) and write one JSON per persona ─────
    per_persona: dict[str, dict[str, str]] = {p: {} for p in to_run}
    truncation_count = 0
    for out, (persona, q) in zip(outputs, keys, strict=True):
        text = out.outputs[0].text
        finish = out.outputs[0].finish_reason
        if finish == "length":
            truncation_count += 1
        per_persona[persona][q] = text

    log.info(
        "Generation done. %d / %d prompts hit length cap (finish_reason=='length').",
        truncation_count,
        len(prompts),
    )

    for persona, qmap in per_persona.items():
        # Per-(persona, q) R-cap assertion: every generated R must be ≤ its
        # per-q cap (vLLM enforces this, but we re-record so the JSON carries
        # the cap that produced it — analyzer sanity).
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

    # vLLM teardown (per .claude/rules/gotchas.md).
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
