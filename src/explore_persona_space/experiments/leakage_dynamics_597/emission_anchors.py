# ruff: noqa: RUF002, RUF003  # research code uses Greek letters, ×, ∪, − and ※ legitimately
"""Phase E (#597) — on-policy emission anchors (vLLM multi-LoRA, sparse steps).

The panel trajectories are teacher-forced (the documented within-condition
dynamics exception); these anchors ground them behaviorally: at the sparse
anchor steps ({20, 40, 100, 200, 400, 528}, BOTH arms — symmetric per the
consistency check), generate the trained model's OWN greedy answers
(``max_new_tokens=2048`` — CLAUDE.md #260 rule) under 4 contexts — the
source, its 2 trained-negative personas, and ``no_persona`` — on the held-out
eval questions, and read marker emission (does the answer contain/terminate
with `` ※``; first emission position logged).

Checkpoint-per-phase: ONE JSON per anchor step persisted the moment it
completes (and skipped on re-run when present).

Run as a SUBPROCESS from the dispatcher (vLLM worker teardown safety).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("issue_597.emission_anchors")

DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_LORA_RANK = 32


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except Exception:
        return "unknown"


def detect_marker_emission(text: str, marker_text: str) -> dict:
    """Pure marker-emission read on one on-policy completion (CPU-testable).

    Substring matching is the ONE sanctioned exception to the no-substring-
    match rule (CLAUDE.md: "Exception: marker leakage only") — the marker is
    an exact single-token string by construction.

    Returns:
        ``{"emitted": bool, "first_pos": int | None, "ends_with": bool,
           "n_occurrences": int}`` where ``first_pos`` is the character index
        of the first marker occurrence.
    """
    first = text.find(marker_text)
    return {
        "emitted": first >= 0,
        "first_pos": first if first >= 0 else None,
        "ends_with": text.rstrip().endswith(marker_text.strip()),
        "n_occurrences": text.count(marker_text),
    }


def build_anchor_prompts(
    tokenizer, contexts: dict[str, str], questions: list[str]
) -> tuple[list[str], list[tuple[str, int]]]:
    """Chat-template prompts for the anchor contexts × questions grid."""
    prompts: list[str] = []
    keys: list[tuple[str, int]] = []
    for name, system_prompt in contexts.items():
        for qi, q in enumerate(questions):
            msgs: list[dict[str, str]] = []
            if system_prompt and system_prompt != "":
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": q})
            prompts.append(
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            )
            keys.append((name, qi))
    return prompts, keys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="#597 Phase E — on-policy emission anchors (vLLM multi-LoRA).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arm", choices=("a", "b"), required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt-root", type=Path, required=True)
    parser.add_argument("--anchor-steps", type=str, required=True, help="Comma-separated steps.")
    parser.add_argument("--eval-pool", type=Path, required=True)
    parser.add_argument(
        "--contexts-json",
        type=str,
        required=True,
        help='JSON object {"name": "system_prompt"} — the 4 anchor contexts '
        "(source + 2 trained negatives + no_persona; empty string = no system msg).",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-lora-rank", type=int, default=DEFAULT_MAX_LORA_RANK)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.leakage_dynamics_597 import (
        BASE_MODEL,
        MARKER_ID,
        MARKER_TEXT,
    )
    from explore_persona_space.experiments.leakage_dynamics_597.probe_rows import (
        load_eval_questions,
    )

    t0 = time.time()
    anchor_steps = [int(s) for s in args.anchor_steps.split(",") if s.strip()]
    contexts: dict[str, str] = json.loads(args.contexts_json)
    if not contexts:
        raise ValueError("--contexts-json decoded to an empty dict")
    questions = load_eval_questions(args.eval_pool, limit=args.limit_questions)
    log.info(
        "[phase=emis_setup_%s_%s] %d anchors x %d contexts x %d questions",
        args.arm,
        args.source,
        len(anchor_steps),
        len(contexts),
        len(questions),
    )

    # Resolve + validate every anchor checkpoint BEFORE paying the vLLM load.
    pending: list[tuple[int, Path, Path]] = []
    for step in anchor_steps:
        ckpt_dir = args.ckpt_root / f"checkpoint-{step}"
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f"anchor checkpoint missing: {ckpt_dir}")
        out_path = args.out_dir / f"{args.source}_step{step:05d}.json"
        if out_path.exists():
            log.info(
                "[phase=emis_anchor_%s_%s] step %d already done (%s); skipping",
                args.arm,
                args.source,
                step,
                out_path,
            )
            continue
        pending.append((step, ckpt_dir, out_path))
    if not pending:
        log.info(
            "[phase=emis_%s_%s] all anchors already persisted; nothing to do", args.arm, args.source
        )
        return 0

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.encode(MARKER_TEXT, add_special_tokens=False) != [MARKER_ID]:
        raise RuntimeError(
            f"marker {MARKER_TEXT!r} -> "
            f"{tokenizer.encode(MARKER_TEXT, add_special_tokens=False)}, expected [{MARKER_ID}]"
        )
    prompts, keys = build_anchor_prompts(tokenizer, contexts, questions)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_loras=1,
    )
    sampling = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for idx, (step, ckpt_dir, out_path) in enumerate(pending):
        t_a = time.time()
        lora_req = LoRARequest(
            lora_name=f"{args.arm}_{args.source}_step{step}",
            lora_int_id=idx + 1,
            lora_path=str(ckpt_dir),
        )
        outputs = llm.generate(prompts, sampling, lora_request=lora_req)
        if len(outputs) != len(prompts):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts")
        rows: list[dict] = []
        for (name, qi), out in zip(keys, outputs, strict=True):
            comp = out.outputs[0]
            emission = detect_marker_emission(comp.text, MARKER_TEXT)
            rows.append(
                {
                    "context": name,
                    "q_idx": qi,
                    "q": questions[qi],
                    "completion": comp.text,
                    "truncated": comp.finish_reason == "length",
                    **emission,
                }
            )
        n_trunc = sum(r["truncated"] for r in rows)
        payload = {
            "schema": "i597_emission_anchor_v1",
            "arm": args.arm,
            "source": args.source,
            "seed": args.seed,
            "step": step,
            "ckpt_dir": str(ckpt_dir),
            "base_model": BASE_MODEL,
            "marker_text": MARKER_TEXT,
            "max_new_tokens": args.max_new_tokens,
            "n_rows": len(rows),
            "truncation_rate": n_trunc / max(1, len(rows)),
            "emission_rate_by_context": {
                name: sum(r["emitted"] for r in rows if r["context"] == name)
                / max(1, sum(1 for r in rows if r["context"] == name))
                for name in contexts
            },
            "rows": rows,
            "metadata": {
                "git_commit": _git_sha(),
                "hostname": socket.gethostname(),
                "ts": datetime.now(UTC).isoformat(),
                "wall_seconds": round(time.time() - t_a, 1),
            },
        }
        tmp = out_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, out_path)
        log.info(
            "[phase=emis_anchor_%s_%s] step %d done in %.1fs (truncation %.3f) -> %s",
            args.arm,
            args.source,
            step,
            time.time() - t_a,
            payload["truncation_rate"],
            out_path,
        )

    log.info(
        "[phase=emis_%s_%s] %d anchor(s) completed in %.1fs",
        args.arm,
        args.source,
        len(pending),
        time.time() - t0,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
