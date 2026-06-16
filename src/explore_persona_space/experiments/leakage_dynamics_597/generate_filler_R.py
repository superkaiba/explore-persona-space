# ruff: noqa: RUF003  # research code uses Greek letters, ×, ∪, − and ※ legitimately
"""Generate per-source filler R (#597 filler arm, plan v5 §2): base-model greedy
responses under the SOURCE persona on the disjoint ``filler_500`` questions.

For each filler question, generate ONE greedy (temp=0) base-Qwen response under
the cell's SOURCE persona system prompt, capped at 1024 new tokens — EXACTLY the
construction the #480 contrastive negatives' R used (on-policy base response
under the row's system prompt, marker-less). The marker is NEVER appended: the
filler rows are marker-less, so their only loss-bearing token under
``MarkerOnlyDataCollator(tail_tokens=0)`` is the EOS (the contrastive-negative
loss surface, plan v5 §2).

Run as a SUBPROCESS from the dispatcher (vLLM worker teardown safety —
gotchas.md):
    uv run python -m explore_persona_space.experiments.leakage_dynamics_597\
.generate_filler_R --source villain --filler-questions data/issue_597/filler/filler_500.jsonl \
        --out-path data/issue_597/filler/villain_filler_R.json

Output JSON (checkpoint-per-phase: written atomically once, immutable after):
    {
      "schema": "i597_filler_R_v1",
      "source": ..., "source_system_prompt": ..., "base_model": ...,
      "marker_text": ..., "max_new_tokens": ..., "n_questions": <N>,
      "truncation_rate": <float>,
      "rows": [{"q": ..., "r_base": ..., "truncated": bool}],
      "metadata": {git_commit, hostname, ts, vllm_version}
    }
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

log = logging.getLogger("issue_597.generate_filler_R")

DEFAULT_MAX_NEW_TOKENS = 1024


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


def load_filler_questions(path: Path, limit: int | None = None) -> list[str]:
    """Load the ``wrong_claim`` strings from the filler_500 JSONL."""
    out: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            wc = obj["wrong_claim"]
            if not isinstance(wc, str):
                raise ValueError(f"malformed filler row: {obj}")
            out.append(wc)
    if not out:
        raise ValueError(f"filler question pool {path} has 0 questions")
    if limit is not None:
        out = out[:limit]
    return out


def render_source_prompts(tokenizer, source_system_prompt: str, questions: list[str]) -> list[str]:
    """Chat-template each (source persona, question) prompt (system role injection)."""
    prompts: list[str] = []
    for q in questions:
        msgs = [
            {"role": "system", "content": source_system_prompt},
            {"role": "user", "content": q},
        ]
        prompts.append(
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )
    return prompts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="#597 filler R generation (vLLM, base model greedy, source persona).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--filler-questions", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--limit-questions",
        type=int,
        default=None,
        help="Smoke-scale knob: cap the question count (threads the dispatcher's "
        "--smoke subset through this phase).",
    )
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
        SOURCE_PERSONAS,
    )
    from explore_persona_space.experiments.marker_implant_480.build_training_pool import (
        SOURCE_SYSTEM_PROMPTS,
    )

    if args.source not in SOURCE_PERSONAS:
        raise ValueError(f"source {args.source} not in SOURCE_PERSONAS {SOURCE_PERSONAS}")
    source_system_prompt = SOURCE_SYSTEM_PROMPTS[args.source]

    t0 = time.time()
    questions = load_filler_questions(args.filler_questions, limit=args.limit_questions)
    log.info(
        "[phase=filler_r_render] source=%s, %d filler questions under the source persona",
        args.source,
        len(questions),
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # In-process marker assert (#537: every process fails at startup on a
    # wrong marker, not just the dispatcher).
    if tokenizer.encode(MARKER_TEXT, add_special_tokens=False) != [MARKER_ID]:
        raise RuntimeError(
            f"marker {MARKER_TEXT!r} -> "
            f"{tokenizer.encode(MARKER_TEXT, add_special_tokens=False)}, expected [{MARKER_ID}]"
        )
    prompts = render_source_prompts(tokenizer, source_system_prompt, questions)

    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
    )
    sampling = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    log.info("[phase=filler_r_generate] generating %d greedy responses...", len(prompts))
    outputs = llm.generate(prompts, sampling)
    if len(outputs) != len(prompts):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts")

    rows: list[dict] = []
    n_truncated = 0
    n_marker_hits = 0
    for q, out in zip(questions, outputs, strict=True):
        comp = out.outputs[0]
        truncated = comp.finish_reason == "length"
        n_truncated += int(truncated)
        # The marker is a rare implanted token — the base prior is ~-21 nat, so
        # a base greedy hit is essentially impossible. Count it for the audit;
        # build_filler_pool fails loud if any R carries the marker.
        if MARKER_TEXT in comp.text:
            n_marker_hits += 1
        rows.append({"q": q, "r_base": comp.text, "truncated": truncated})
    truncation_rate = n_truncated / max(1, len(prompts))
    log.info(
        "[phase=filler_r_generate] done: %d responses, truncation_rate=%.4f, marker_hits=%d",
        len(prompts),
        truncation_rate,
        n_marker_hits,
    )
    if n_marker_hits:
        # Not silently dropped — surfaced loudly so the operator sees a base-model
        # marker leak before build_filler_pool's per-row assert fires.
        log.warning(
            "[phase=filler_r_generate] %d base greedy R contain the marker %r — "
            "build_filler_pool will FAIL on these (the base prior should be ~0).",
            n_marker_hits,
            MARKER_TEXT,
        )

    payload = {
        "schema": "i597_filler_R_v1",
        "source": args.source,
        "source_system_prompt": source_system_prompt,
        "base_model": BASE_MODEL,
        "marker_text": MARKER_TEXT,
        "max_new_tokens": args.max_new_tokens,
        "n_questions": len(questions),
        "truncation_rate": truncation_rate,
        "n_marker_hits": n_marker_hits,
        "rows": rows,
        "metadata": {
            "git_commit": _git_sha(),
            "hostname": socket.gethostname(),
            "ts": datetime.now(UTC).isoformat(),
            "vllm_version": vllm.__version__,
            "wall_seconds": round(time.time() - t0, 1),
        },
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, args.out_path)
    log.info("[phase=filler_r_write] filler R -> %s", args.out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
