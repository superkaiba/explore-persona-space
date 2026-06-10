# ruff: noqa: RUF002, RUF003  # research code uses Greek letters (ρ, Δ), × and − legitimately
"""Task #480 Phase 2a — vLLM batched on-policy R_trained generation.

For ONE trained source (merged model dir), iterate over the 24 panel
personas × 50 held-out Q_eval questions and generate ONE greedy (temp=0)
response R_trained per (panel, q) — the trained model's own answer under
each panel persona's system prompt.

Subprocess-isolated from Phase 2b (HF Transformers logprob) per gotchas.md
vLLM-teardown bug: vLLM TP worker subprocesses survive in-process
``destroy_*`` calls and re-grab GPU memory the moment HF Transformers
loads. The cleanest hammer is to write R_trained to JSON, exit, and let
the kernel reap vLLM children before Phase 2b loads anything.

Output:

    eval_results/issue_480/per_source/<source>/seed_42/r_trained.json

  { "source", "seed", "n_panel", "n_questions", "panel_personas",
    "r_trained": { "<panel>": [str, ...] },   # len == n_questions per panel
    "questions": [str, ...],
    "merged_model_path", "git_commit_sha", "timestamp_utc", "hostname" }

  Also writes per-panel ``raw_completions/<panel>_seed<S>.json`` for the
  upload-policy raw-completions contract.

Why max_new_tokens=2048: marker / end-of-completion eval per CLAUDE.md
critical rule (≥ 2× longest trained completion). Truncating R_trained
creates a R-tail that doesn't carry the marker → silent zero on the DV.
"""

from __future__ import annotations

import argparse
import gc
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

log = logging.getLogger("issue_480.phase2a")

DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0
SEED = 42


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


def _load_questions(path: Path) -> list[str]:
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
    return out


def _build_prompt(tokenizer, system_prompt: str, user_text: str) -> str:
    msgs: list[dict[str, str]] = []
    if system_prompt and system_prompt != "":
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_text})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--merged-model-path", type=Path, required=True)
    parser.add_argument("--eval-pool", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--sentinel-path",
        type=Path,
        default=Path("/workspace/logs/issue-480-phase2a-results.json"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw_completions"
    raw_dir.mkdir(parents=True, exist_ok=True)

    questions = _load_questions(args.eval_pool)
    log.info(
        "[phase=phase2a] source=%s n_questions=%d merged=%s",
        args.source,
        len(questions),
        args.merged_model_path,
    )

    # Lazy import vLLM / panel.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    if args.source not in EVAL_PERSONAS_24:
        raise ValueError(
            f"source {args.source} not in EVAL_PERSONAS_24 ({sorted(EVAL_PERSONAS_24.keys())})"
        )

    panel_personas = sorted(EVAL_PERSONAS_24.keys())
    log.info("[phase=phase2a] panel=%d personas", len(panel_personas))

    tokenizer = AutoTokenizer.from_pretrained(str(args.merged_model_path))

    log.info("[phase=phase2a] Loading merged model into vLLM ...")
    llm = LLM(
        model=str(args.merged_model_path),
        seed=args.seed,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
    )

    sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=1.0,
        seed=args.seed,
        max_tokens=args.max_new_tokens,
    )

    r_trained: dict[str, list[str]] = {}
    t_phase = time.time()

    for panel in panel_personas:
        system_prompt = EVAL_PERSONAS_24[panel]
        prompts = [_build_prompt(tokenizer, system_prompt, q) for q in questions]
        t_panel = time.time()
        outs = llm.generate(prompts, sampling)
        responses = [o.outputs[0].text for o in outs]
        r_trained[panel] = responses

        # Per-panel raw completions (upload-policy contract).
        raw_path = raw_dir / f"{panel}_seed{args.seed}.json"
        with open(raw_path, "w") as f:
            json.dump(
                {
                    "source": args.source,
                    "panel_persona": panel,
                    "seed": args.seed,
                    "merged_model_path": str(args.merged_model_path),
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "rows": [
                        {"q": q, "response": r} for q, r in zip(questions, responses, strict=True)
                    ],
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                },
                f,
                ensure_ascii=False,
            )
        log.info(
            "[phase=phase2a] panel=%s wall=%.1fs raw -> %s",
            panel,
            time.time() - t_panel,
            raw_path,
        )

    # Aggregate JSON for Phase 2b.
    out_path = args.out_dir / "r_trained.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "source": args.source,
                "seed": args.seed,
                "n_panel": len(panel_personas),
                "n_questions": len(questions),
                "panel_personas": panel_personas,
                "panel_system_prompts": {p: EVAL_PERSONAS_24[p] for p in panel_personas},
                "questions": questions,
                "r_trained": r_trained,
                "merged_model_path": str(args.merged_model_path),
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "git_commit_sha": _git_sha(),
                "hostname": socket.gethostname(),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            },
            f,
            ensure_ascii=False,
        )
    log.info("[phase=phase2a] aggregate r_trained -> %s", out_path)

    # vLLM teardown — best-effort (the real safety is subprocess exit + kernel reap).
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        del llm
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        log.warning("[phase=phase2a] vLLM destroy failed (non-fatal): %s", e)
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    wall = time.time() - t_phase
    sentinel = {
        "phase": "phase2a_generate_R_trained",
        "issue": 480,
        "source": args.source,
        "seed": args.seed,
        "n_panel": len(panel_personas),
        "n_questions": len(questions),
        "out_path": str(out_path),
        "raw_dir": str(raw_dir),
        "wall_seconds": round(wall, 1),
        "git_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    args.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.sentinel_path, "w") as f:
        json.dump(sentinel, f, indent=2)
    log.info("[phase=phase2a] DONE wall=%.1fs sentinel=%s", wall, args.sentinel_path)
    print("[phase=done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
