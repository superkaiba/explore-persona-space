"""Task #411 Phase 2 — vLLM batched sycophancy eval for one trained source.

For one merged Qwen-7B checkpoint (one source's LoRA, already merged to disk),
generates 10 free-form completions per (24 panel personas x 50 held-out wrong
claims) and writes per-panel JSONs.

Workload per source:
    24 panel personas x 50 claims x 10 rollouts = 12,000 generations.

vLLM is loaded ONCE per call to eval_source() and re-used across all 24 panel
personas (the prompts only differ by the system-prompt prefix, so prefix
caching is helpful but not load-bearing). This matches the #391 persona-batching
pattern.

Outputs per source under ``<slab_root>/<source>/seed_<seed>/``:

    sycophancy_eval_<panel_persona>.json
        - aggregated metadata + ALL 500 completions for one panel persona
    raw_completions/<panel_persona>_seed{seed}.json
        - same completions in the canonical "raw_completions" shape so the
          analyzer's cherry-picked-sample rule has a permanent URL.

Why max_new_tokens=512 (not 2048):
    Free-generation behavioral eval (no end-of-completion marker, no
    truncation-silent-zero risk). Plan §4 Phase 2 paragraph "Why 512
    max_tokens (not 2048)". CLAUDE.md max_new_tokens >=2048 rule applies to
    marker / end-of-completion evals; this one is in the exception class.

Sentinel write after the source completes:
    /workspace/logs/issue-411-<source>-results.json
        {"source": ..., "phase": "eval_complete", "n_panel_jsons": 24, ...}
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

log = logging.getLogger("issue_411.eval_one_source")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_N_ROLLOUTS = 10
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 1.0


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _load_eval_pool(eval_pool_path: Path) -> list[dict[str, str]]:
    """Read held-out wrong claims from train_pool JSONL."""
    if not eval_pool_path.exists():
        raise FileNotFoundError(
            f"Eval pool {eval_pool_path} missing — run Phase 0 first to "
            f"materialize the held-out claims."
        )
    out: list[dict[str, str]] = []
    with open(eval_pool_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append({"wrong_claim": obj["wrong_claim"], "correction": obj["correction"]})
    return out


def _build_prompt_text(tokenizer, panel_prompt: str, wrong_claim: str) -> str:
    """Apply Qwen2.5-Instruct chat template -> single string for vLLM."""
    messages = [
        {"role": "system", "content": panel_prompt},
        {"role": "user", "content": wrong_claim},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _persona_jsonl_path(out_dir: Path, panel_persona: str) -> Path:
    return out_dir / f"sycophancy_eval_{panel_persona}.json"


def _raw_completions_path(out_dir: Path, panel_persona: str, seed: int) -> Path:
    return out_dir / "raw_completions" / f"{panel_persona}_seed{seed}.json"


def _write_panel_outputs(
    out_dir: Path,
    *,
    source: str,
    seed: int,
    panel_persona: str,
    panel_prompt: str,
    claims: list[dict[str, str]],
    completions: list[list[str]],
    metadata: dict,
) -> None:
    """Write the per-panel eval JSON AND the canonical raw_completions JSON.

    Both files are produced atomically per panel persona so a downstream crash
    on the NEXT panel persona doesn't lose this one (CLAUDE.md
    "Checkpoint per phase").
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_completions").mkdir(parents=True, exist_ok=True)
    flat_records: list[dict] = []
    for c_idx, (claim, c_rollouts) in enumerate(zip(claims, completions, strict=True)):
        for r_idx, completion in enumerate(c_rollouts):
            flat_records.append(
                {
                    "claim": claim["wrong_claim"],
                    "correction": claim["correction"],
                    "claim_idx": c_idx,
                    "rollout_idx": r_idx,
                    "completion": completion,
                }
            )
    payload = {
        "source": source,
        "seed": seed,
        "panel_persona": panel_persona,
        "panel_prompt": panel_prompt,
        "n_claims": len(claims),
        "n_rollouts_per_claim": len(completions[0]) if completions else 0,
        "completions": flat_records,
        "metadata": metadata,
    }
    persona_path = _persona_jsonl_path(out_dir, panel_persona)
    with open(persona_path, "w") as f:
        json.dump(payload, f)

    # Mirror to canonical raw_completions/<panel>_seed{S}.json shape so the
    # analyzer can reference each panel persona's raw text without parsing
    # the aggregated eval JSON.
    raw_path = _raw_completions_path(out_dir, panel_persona, seed)
    with open(raw_path, "w") as f:
        json.dump(payload, f)


def eval_source(
    *,
    source: str,
    seed: int,
    merged_model_path: Path | None,
    eval_pool_path: Path,
    out_dir: Path,
    n_rollouts: int = DEFAULT_N_ROLLOUTS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    gpu_memory_utilization: float = 0.85,
    hub_model_id: str | None = None,
) -> dict[str, object]:
    """Run the full 24-panel x 50-claim x N-rollout eval for one source.

    Loads vLLM ONCE; loops over the 24 panel personas inside the same LLM
    instance (call ``LLM.generate(...)`` per panel persona for clean isolation
    of system prompts).

    Args:
        merged_model_path: Local merged Qwen+LoRA dir from Phase 1. Mutually
            exclusive with ``hub_model_id``; provide exactly one.
        hub_model_id: HF Hub model id (e.g. ``Qwen/Qwen2.5-7B-Instruct``) for
            the base-panel baseline pass (Phase 3 step 2 in the plan). When
            set, ``merged_model_path`` is ignored and vLLM loads the base
            model directly from the Hub (or the HF cache). No LoRA applied.

    Returns a summary dict that's also written to ``eval_summary.json`` in
    ``out_dir``.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("eval_source(source=%s, seed=%d, out_dir=%s)", source, seed, out_dir)

    if (merged_model_path is None) == (hub_model_id is None):
        raise ValueError(
            "Provide EXACTLY ONE of merged_model_path / hub_model_id. "
            f"Got merged_model_path={merged_model_path}, hub_model_id={hub_model_id}."
        )

    if merged_model_path is not None and not merged_model_path.exists():
        raise FileNotFoundError(
            f"Merged model dir not found: {merged_model_path}. Phase 1 must "
            f"have completed and merged before eval can run."
        )

    claims = _load_eval_pool(eval_pool_path)
    log.info("Loaded %d held-out wrong claims from %s", len(claims), eval_pool_path)
    if len(claims) == 0:
        raise ValueError(f"Eval pool {eval_pool_path} contained zero claims.")

    model_arg = str(merged_model_path) if merged_model_path is not None else hub_model_id
    tokenizer = AutoTokenizer.from_pretrained(
        model_arg, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    log.info("Loading vLLM on %s ...", model_arg)
    t_load_start = time.time()
    llm = LLM(
        model=model_arg,
        tensor_parallel_size=1,
        max_model_len=2048,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    log.info("vLLM loaded in %.1fs", time.time() - t_load_start)

    sampling = SamplingParams(
        n=n_rollouts,
        temperature=temperature,
        max_tokens=max_new_tokens,
        seed=seed,
    )

    panel_summaries: dict[str, dict] = {}
    t_start = time.time()
    for panel_idx, (panel_persona, panel_prompt) in enumerate(EVAL_PERSONAS_24.items(), 1):
        log.info(
            "[%d/%d] panel_persona=%s — generating %d prompts x %d rollouts ...",
            panel_idx,
            len(EVAL_PERSONAS_24),
            panel_persona,
            len(claims),
            n_rollouts,
        )
        prompts = [_build_prompt_text(tokenizer, panel_prompt, c["wrong_claim"]) for c in claims]

        t_panel_start = time.time()
        outputs = llm.generate(prompts, sampling)
        t_panel = time.time() - t_panel_start
        if len(outputs) != len(claims):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(claims)} prompts "
                f"(panel={panel_persona})"
            )

        # Each request output has a `.outputs` list of length n_rollouts.
        completions: list[list[str]] = []
        for req_out in outputs:
            rollouts = [o.text for o in req_out.outputs]
            if len(rollouts) != n_rollouts:
                raise RuntimeError(
                    f"Expected {n_rollouts} rollouts per claim, got {len(rollouts)} "
                    f"for panel={panel_persona}"
                )
            completions.append(rollouts)

        panel_meta = {
            "source": source,
            "seed": seed,
            "panel_persona": panel_persona,
            "n_claims": len(claims),
            "n_rollouts_per_claim": n_rollouts,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "merged_model_path": (
                str(merged_model_path) if merged_model_path is not None else None
            ),
            "hub_model_id": hub_model_id,
            "model_loaded": model_arg,
            "base_model": BASE_MODEL,
            "git_commit_sha": _git_sha(),
            "hostname": socket.gethostname(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "gen_wall_seconds": round(t_panel, 1),
        }
        _write_panel_outputs(
            out_dir,
            source=source,
            seed=seed,
            panel_persona=panel_persona,
            panel_prompt=panel_prompt,
            claims=claims,
            completions=completions,
            metadata=panel_meta,
        )
        panel_summaries[panel_persona] = {
            "n_completions": len(claims) * n_rollouts,
            "wall_seconds": round(t_panel, 1),
        }
        log.info(
            "panel=%s done in %.1fs (%.1f gen/s)",
            panel_persona,
            t_panel,
            (len(claims) * n_rollouts) / max(t_panel, 1e-6),
        )

    wall = time.time() - t_start
    summary = {
        "source": source,
        "seed": seed,
        "n_panel_personas": len(panel_summaries),
        "n_claims_per_panel": len(claims),
        "n_rollouts_per_claim": n_rollouts,
        "total_completions": sum(p["n_completions"] for p in panel_summaries.values()),
        "wall_seconds": round(wall, 1),
        "merged_model_path": (str(merged_model_path) if merged_model_path is not None else None),
        "hub_model_id": hub_model_id,
        "model_loaded": model_arg,
        "base_model": BASE_MODEL,
        "git_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "per_panel": panel_summaries,
    }
    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Free vLLM workers AGGRESSIVELY before returning. See CLAUDE.md gotcha
    # on vLLM teardown leaving orphan worker subprocesses. Even though we
    # don't immediately load HF Transformers in this same process after,
    # the dispatcher may chain another phase; subprocess isolation in the
    # dispatcher is the recommended belt-AND-suspenders.
    del llm
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        log.warning("vLLM destroy_* failed: %s (continuing)", e)
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    return summary


def _write_sentinel(source: str, summary: dict, sentinel_path: Path) -> None:
    """Write the per-source pod-side sentinel that the orchestrator polls."""
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel = {
        "source": source,
        "phase": "eval_complete",
        "completed_phases": ["train", "merge", "eval"],
        "n_panel_jsons": summary.get("n_panel_personas"),
        "n_completions": summary.get("total_completions"),
        "wall_seconds": summary.get("wall_seconds"),
        "merged_model_path": summary.get("merged_model_path"),
        "git_commit_sha": summary.get("git_commit_sha"),
        "hostname": summary.get("hostname"),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    with open(sentinel_path, "w") as f:
        json.dump(sentinel, f, indent=2)
    log.info("Wrote sentinel to %s", sentinel_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Source persona name (e.g. villain)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--merged-model-path",
        type=Path,
        default=None,
        help=(
            "Path to the merged Qwen-7B + LoRA on disk. Mutually exclusive "
            "with --hub-model-id; provide exactly one."
        ),
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help=(
            "HF Hub model id (e.g. Qwen/Qwen2.5-7B-Instruct) for the base-panel "
            "baseline pass (plan §4 Phase 3 step 2). When set, --merged-model-path "
            "must be omitted; vLLM loads the model from the Hub / HF cache and "
            "no LoRA is applied."
        ),
    )
    parser.add_argument(
        "--eval-pool",
        type=Path,
        required=True,
        help="Path to data/issue_411/wrong_claims/eval_50.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Where per-panel eval JSONs go: <slab_root>/<source>/seed_<seed>/",
    )
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=DEFAULT_N_ROLLOUTS,
        help=f"Rollouts per (panel, claim) pair (default {DEFAULT_N_ROLLOUTS})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Generation cap (default {DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM GPU memory utilization (default 0.85).",
    )
    parser.add_argument(
        "--sentinel-path",
        type=Path,
        default=None,
        help="Where to write the per-source pod-side sentinel JSON. If omitted, "
        "defaults to /workspace/logs/issue-411-<source>-results.json.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=phase2] %(message)s")

    summary = eval_source(
        source=args.source,
        seed=args.seed,
        merged_model_path=args.merged_model_path,
        eval_pool_path=args.eval_pool,
        out_dir=args.out_dir,
        n_rollouts=args.n_rollouts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
        hub_model_id=args.hub_model_id,
    )

    sentinel_path = args.sentinel_path or Path(
        f"/workspace/logs/issue-411-{args.source}-results.json"
    )
    _write_sentinel(args.source, summary, sentinel_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
