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
completes, and skipped on re-run ONLY when its stored ladder run-id matches
the on-disk ladder's (same ``resolve_ladder_run_id`` contract as
``panel_probe`` — Arm A: the immutable-HF literal; Arm B: the dispatcher's
``ladder_run_id.json``). A missing or mismatched run-id means the anchors
were generated against a DIFFERENT training run's weights (end-of-cell
cleanup rmtrees the ladder, so a later-cell crash + relaunch RETRAINS this
cell under a fresh run-id) and the step is re-anchored (overwritten).

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


def stored_anchor_is_current(stored_payload: dict, ladder_run_id: str) -> bool:
    """True iff a stored anchor JSON was generated against the CURRENT ladder.

    Same contract as ``panel_probe.stored_probe_is_current`` (#597 round-4
    fix 2): a missing ``ladder_run_id`` key (anchors written before provenance
    was threaded — the legacy ``i597_emission_anchor_v1`` shape) counts as a
    MISMATCH, because the read cannot be attributed to the on-disk weights.
    """
    return stored_payload.get("ladder_run_id") == ladder_run_id


def resolve_pending_anchors(
    anchor_steps: list[int],
    ckpt_root: Path,
    out_dir: Path,
    source: str,
    arm: str,
    ladder_run_id: str,
) -> list[tuple[int, Path, Path]]:
    """Validate every anchor checkpoint, then drop steps already persisted
    against the CURRENT ladder (run-id resume gate; CPU-testable, no vLLM).

    A stored anchor with a missing or mismatched ``ladder_run_id`` was
    generated against a DIFFERENT training run's weights (end-of-cell cleanup
    rmtrees the ladder, so a later-cell crash + relaunch RETRAINS this cell
    under a fresh run-id while the stale anchors survive on disk) — it is
    returned as pending and overwritten, never silently trusted.

    Returns:
        ``[(step, ckpt_dir, out_path), ...]`` for the steps still to run.
    """
    pending: list[tuple[int, Path, Path]] = []
    for step in anchor_steps:
        ckpt_dir = ckpt_root / f"checkpoint-{step}"
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f"anchor checkpoint missing: {ckpt_dir}")
        out_path = out_dir / f"{source}_step{step:05d}.json"
        if out_path.exists():
            with open(out_path) as f:
                stored = json.load(f)
            if stored_anchor_is_current(stored, ladder_run_id):
                log.info(
                    "[phase=emis_anchor_%s_%s] step %d already done "
                    "(ladder run-id match; %s); skipping",
                    arm,
                    source,
                    step,
                    out_path,
                )
                continue
            log.warning(
                "[phase=emis_anchor_%s_%s] step %d stored anchors are STALE (stored ladder "
                "run-id %r != current %r) — regenerating and overwriting %s",
                arm,
                source,
                step,
                stored.get("ladder_run_id"),
                ladder_run_id,
                out_path,
            )
        pending.append((step, ckpt_dir, out_path))
    return pending


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
    # 0.60 (not 0.85): vLLM init probes free VRAM the instant the prior HF
    # subprocess exits, before the driver finishes reclaiming (~16 GiB residue
    # observed on all 4 shards, 2026-06-11); 0.60*80 GiB = 47.5 GiB clears even
    # with one stale 7B footprint still resident.
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
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
    from explore_persona_space.experiments.leakage_dynamics_597.panel_probe import (
        resolve_ladder_run_id,
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

    # Provenance BEFORE the vLLM load (fail fast on a missing Arm B id), then
    # resolve + validate every anchor checkpoint with the run-id resume gate.
    ladder_run_id = resolve_ladder_run_id(args.arm, args.ckpt_root)
    log.info(
        "[phase=emis_setup_%s_%s] ladder run-id: %s",
        args.arm,
        args.source,
        ladder_run_id,
    )
    pending = resolve_pending_anchors(
        anchor_steps, args.ckpt_root, args.out_dir, args.source, args.arm, ladder_run_id
    )
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
            "ladder_run_id": ladder_run_id,
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
