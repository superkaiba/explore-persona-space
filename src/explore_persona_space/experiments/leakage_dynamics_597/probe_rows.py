# ruff: noqa: RUF002, RUF003  # research code uses Greek letters, ×, ∪, − and ※ legitimately
"""Phase 0 (#597) — fixed probe-row generation: base on-policy R per (context, question).

For each of the 25 probe contexts (the 24-persona eval panel + the bare
``no_persona`` chat) × the 50 held-out ``eval_50`` wrong-claim questions,
generate ONE greedy (temp=0) base-model response, capped at 1024 new tokens
(plan §11: R-cap 1024 per the measurement recipe step 1). These rows are
FIXED for the whole experiment — every checkpoint of BOTH arms is probed on
the same rows, which is what makes the trajectories step-comparable.

Run as a SUBPROCESS from the dispatcher (vLLM worker teardown safety —
gotchas.md): ``uv run python -m
explore_persona_space.experiments.leakage_dynamics_597.probe_rows ...``.

Output JSON (checkpoint-per-phase: written atomically once, then treated as
immutable input by every later phase):

    {
      "schema": "i597_probe_rows_v1",
      "base_model": ..., "marker_text": ..., "max_new_tokens": ...,
      "n_contexts": 25, "n_questions": <N>,
      "truncation_rate": <float>,
      "contexts": {
        "<name>": {"system_prompt": "<str, '' for no_persona>",
                    "rows": [{"q": ..., "r_base": ..., "truncated": bool}]}
      },
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

log = logging.getLogger("issue_597.probe_rows")

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


def load_eval_questions(path: Path, limit: int | None = None) -> list[str]:
    """Load the ``wrong_claim`` strings from an eval-pool JSONL (#411 schema)."""
    out: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            wc = obj["wrong_claim"]
            if not isinstance(wc, str):
                raise ValueError(f"malformed eval row: {obj}")
            out.append(wc)
    if not out:
        raise ValueError(f"eval pool {path} has 0 questions")
    if limit is not None:
        out = out[:limit]
    return out


def render_prompts(
    tokenizer, contexts: dict[str, str], questions: list[str]
) -> tuple[list[str], list[tuple[str, int]]]:
    """Chat-template prompts for every (context, question) cell.

    Persona injection ALWAYS via the system role; the ``no_persona`` context
    (empty-string system prompt) emits NO system message — matching the #480
    prompt builders (``i480_phase2a_generate_R_trained._build_prompt``).

    Returns:
        ``(prompts, keys)`` with ``keys[i] = (context_name, question_idx)``.
    """
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
    assert len(prompts) == len(contexts) * len(questions), (len(prompts), len(contexts))
    return prompts, keys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="#597 Phase 0 probe-row generation (vLLM, base model greedy).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--eval-pool", type=Path, required=True)
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
        probe_contexts_25,
    )

    t0 = time.time()
    contexts = probe_contexts_25()
    questions = load_eval_questions(args.eval_pool, limit=args.limit_questions)
    log.info(
        "[phase=p0_render] %d contexts x %d questions = %d prompts",
        len(contexts),
        len(questions),
        len(contexts) * len(questions),
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
    prompts, keys = render_prompts(tokenizer, contexts, questions)

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
    log.info("[phase=p0_generate] generating %d greedy responses...", len(prompts))
    outputs = llm.generate(prompts, sampling)
    if len(outputs) != len(prompts):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts")

    per_context: dict[str, dict] = {
        name: {"system_prompt": sp, "rows": [{} for _ in questions]}
        for name, sp in contexts.items()
    }
    n_truncated = 0
    for (name, qi), out in zip(keys, outputs, strict=True):
        comp = out.outputs[0]
        truncated = comp.finish_reason == "length"
        n_truncated += int(truncated)
        per_context[name]["rows"][qi] = {
            "q": questions[qi],
            "r_base": comp.text,
            "truncated": truncated,
        }
    truncation_rate = n_truncated / max(1, len(prompts))
    log.info(
        "[phase=p0_generate] done: %d responses, truncation_rate=%.4f (%d truncated)",
        len(prompts),
        truncation_rate,
        n_truncated,
    )

    payload = {
        "schema": "i597_probe_rows_v1",
        "base_model": BASE_MODEL,
        "marker_text": MARKER_TEXT,
        "max_new_tokens": args.max_new_tokens,
        "n_contexts": len(contexts),
        "n_questions": len(questions),
        "truncation_rate": truncation_rate,
        "contexts": per_context,
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
    log.info("[phase=p0_write] probe rows -> %s", args.out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
