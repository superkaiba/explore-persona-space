#!/usr/bin/env python3
"""Task #411 dispatcher — UNIFIED smoke = sweep with one source.

Per-cell discipline (one cell == one source persona, sequential on 1x H100):

  1. Build training pool (CPU) via build_training_pool.build_training_pool(source).
  2. Train LoRA via explore_persona_space.train.sft.train_lora (#99 hparams).
  3. Merge LoRA into base via explore_persona_space.train.sft.merge_lora.
  4. Subprocess-isolate Phase 2 eval (vLLM teardown OOM risk per CLAUDE.md
     gotcha: orphan vLLM workers re-allocate freed GPU memory the moment a
     second framework loads. Easiest hammer: spawn a fresh subprocess per
     source for the vLLM Phase 2.).
  5. Upload merged adapter to HF Hub model repo.
  6. shutil.rmtree(output_dir/'merged') before the next source.
  7. Pod writes per-source sentinel JSON to /workspace/logs/issue-411-<source>-results.json.

Smoke = sweep with one source: pass --only-source villain (or --sources villain
explicitly). Same per-cell function path; no diverging code between smoke + sweep.

Pod-side discipline:
    - Sets EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1 in child env before each cell.
    - NEVER calls scripts/task.py (pods run on issue-<N> branches; task.py
      branch-guards to main). Sentinel-file pattern only.
    - Every subprocess.* call uses env={**os.environ}; load_dotenv() is at
      module-top so HF_TOKEN / WANDB_API_KEY / ANTHROPIC_API_KEY are present
      in os.environ before any subprocess is spawned.

End-of-sweep sentinel:
    /workspace/logs/issue-411-results.json — written ONLY when every
    requested source has its per-source sentinel on disk. Includes
    `epm:results v1`-shaped payload (eval numbers / paths / repro card /
    deviations / final commit SHA).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("dispatch_sycophancy_411")

DEFAULT_SOURCES = (
    "villain",
    "comedian",
    "assistant",
    "qwen_default",
    "software_engineer",
    "kindergarten_teacher",
)
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SEED = 42
HF_REPO = "superkaiba1/explore-persona-space"


def _parse_sources(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},  # explicit env
        ).strip()
    except Exception:
        return "unknown"


def _build_training_pool(source: str, train_pool_path: Path, out_path: Path) -> None:
    """Phase 1 data prep — in-process call (CPU, no GPU concerns)."""
    from explore_persona_space.experiments.sycophancy_implantation_411.build_training_pool import (
        build_training_pool,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_training_pool(source=source, train_pool_path=train_pool_path, output_path=out_path)


def _train_and_merge(
    source: str, seed: int, train_jsonl: Path, output_dir: Path
) -> tuple[Path, Path]:
    """Phase 1 train + merge in-process. Returns (adapter_dir, merged_dir)."""
    from explore_persona_space.train.sft import TrainLoraConfig, merge_lora, train_lora

    adapter_dir = output_dir / "adapter"
    merged_dir = output_dir / "merged"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # #99 verbatim hparams.
    cfg = TrainLoraConfig(
        gpu_id=0,
        epochs=3,
        lr=1e-5,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        batch_size=4,
        grad_accum=4,  # effective batch 16
        max_length=1024,
        warmup_ratio=0.05,
        seed=seed,
        run_name=f"issue411_{source}_seed{seed}",
        report_to="wandb",
        save_strategy="no",
        gradient_checkpointing=True,
        packing=False,
        hf_upload=True,
        hf_repo=HF_REPO,
        hf_path_in_repo=f"adapters/issue_411/{source}_seed{seed}",
    )
    log.info("[%s] Training LoRA -> %s", source, adapter_dir)
    train_lora(
        base_model_path=BASE_MODEL,
        data_path=str(train_jsonl),
        output_dir=str(adapter_dir),
        cfg=cfg,
    )

    log.info("[%s] Merging LoRA into base -> %s", source, merged_dir)
    merge_lora(
        base_model_path=BASE_MODEL,
        adapter_path=str(adapter_dir),
        output_dir=str(merged_dir),
        gpu_id=0,
    )
    return adapter_dir, merged_dir


def _eval_subprocess(
    source: str,
    seed: int,
    merged_dir: Path,
    eval_pool: Path,
    eval_out_dir: Path,
    sentinel_path: Path,
) -> None:
    """Spawn the Phase 2 vLLM eval in a fresh subprocess (vLLM teardown safety).

    Inheriting os.environ via env= is REQUIRED so HF_TOKEN / WANDB_API_KEY
    flow through. Failure here propagates loudly (no try/except: pass).
    """
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "explore_persona_space.experiments.sycophancy_implantation_411.eval_one_source",
        "--source",
        source,
        "--seed",
        str(seed),
        "--merged-model-path",
        str(merged_dir),
        "--eval-pool",
        str(eval_pool),
        "--out-dir",
        str(eval_out_dir),
        "--sentinel-path",
        str(sentinel_path),
    ]
    log.info("[%s] Spawning eval subprocess: %s", source, " ".join(cmd))
    env = {**os.environ}
    env.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")
    subprocess.run(cmd, env=env, check=True)


def _run_one_cell(
    source: str,
    seed: int,
    train_pool: Path,
    eval_pool: Path,
    slab_root: Path,
    runs_root: Path,
) -> dict[str, object]:
    """Train -> merge -> eval -> upload-adapter -> rmtree merged, for one source.

    Sentinel path is /workspace/logs/issue-411-<source>-results.json.
    """
    t_start = time.time()
    output_dir = runs_root / f"{source}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_out_dir = slab_root / source / f"seed_{seed}"
    sentinel_path = Path(f"/workspace/logs/issue-411-{source}-results.json")
    train_jsonl = output_dir / "train_pool.jsonl"

    log.info("=" * 70)
    log.info("[%s] CELL START — output_dir=%s, eval_out_dir=%s", source, output_dir, eval_out_dir)

    _build_training_pool(source, train_pool, train_jsonl)
    _, merged_dir = _train_and_merge(source, seed, train_jsonl, output_dir)
    _eval_subprocess(
        source=source,
        seed=seed,
        merged_dir=merged_dir,
        eval_pool=eval_pool,
        eval_out_dir=eval_out_dir,
        sentinel_path=sentinel_path,
    )

    # Note: train_lora(hf_upload=True) auto-uploads the adapter (see
    # TrainLoraConfig defaults). Verify here without re-uploading; loud-fail
    # if the adapter dir is empty (something silently broke upstream).
    adapter_safetensors = list((output_dir / "adapter").glob("*.safetensors"))
    if not adapter_safetensors:
        raise RuntimeError(
            f"[{source}] Adapter dir {output_dir / 'adapter'} has no .safetensors "
            f"files after training — upload may be stale or training silently failed."
        )

    # rmtree merged dir BEFORE the next cell (MooseFS quota discipline).
    if merged_dir.exists():
        log.info("[%s] rmtree(%s) to free MooseFS quota", source, merged_dir)
        shutil.rmtree(merged_dir, ignore_errors=False)

    wall = time.time() - t_start
    log.info("[%s] CELL DONE in %.1fs", source, wall)
    return {
        "source": source,
        "seed": seed,
        "wall_seconds": round(wall, 1),
        "output_dir": str(output_dir),
        "eval_out_dir": str(eval_out_dir),
        "sentinel_path": str(sentinel_path),
        "adapter_hf_path": f"adapters/issue_411/{source}_seed{seed}",
    }


def _write_final_sentinel(
    sources: list[str],
    seed: int,
    slab_root: Path,
    per_cell: list[dict],
    sources_requested: list[str],
    plan_deviations: list[str],
    final_path: Path = Path("/workspace/logs/issue-411-results.json"),
) -> None:
    """End-of-sweep sentinel — orchestrator's poll_pipeline.py converts to
    `epm:results v1`."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    eval_paths: dict[str, str] = {}
    for c in per_cell:
        eval_paths[c["source"]] = c["eval_out_dir"]
    payload = {
        "schema": "epm:results v1",
        "issue": 411,
        "seed": seed,
        "sources_requested": sources_requested,
        "sources_completed": [c["source"] for c in per_cell],
        "n_completed": len(per_cell),
        "n_requested": len(sources_requested),
        "eval_paths": eval_paths,
        "eval_numbers": {
            "n_panel_personas": 24,
            "n_eval_claims": 50,
            "n_rollouts_per_claim": 10,
            "n_total_completions_per_source": 12000,
        },
        "reproducibility_card": {
            "base_model": BASE_MODEL,
            "hf_model_repo": HF_REPO,
            "hf_data_repo": "superkaiba1/explore-persona-space-data",
            "adapter_paths": {
                c["source"]: f"{HF_REPO}/tree/main/{c['adapter_hf_path']}" for c in per_cell
            },
        },
        "worktree_path": (
            "/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-411"
        ),
        "final_commit_sha": _git_sha(),
        "wandb_url": "n/a (per-cell wandb runs; see wandb project=issue411_<source>_seed42)",
        "hf_hub_url": f"https://huggingface.co/{HF_REPO}/tree/main/adapters/issue_411",
        "gpu_hours_used_estimate": round(sum(c["wall_seconds"] for c in per_cell) / 3600, 2),
        "gpu_hours_budgeted": 10.0,
        "plan_deviations": plan_deviations,
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    with open(final_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Final sentinel: %s", final_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        type=_parse_sources,
        default=list(DEFAULT_SOURCES),
        help="Comma-separated source personas (default: all 6).",
    )
    parser.add_argument(
        "--only-source",
        type=str,
        default=None,
        help=(
            "Single source persona to run (smoke shortcut). When set, "
            "OVERRIDES --sources. Equivalent to `--sources <name>`."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Shorthand for --only-source villain. Smoke runs the exact same "
        "per-cell function as the sweep; the only diff is set size.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--train-pool",
        type=Path,
        default=Path("data/issue_411/wrong_claims/train_200.jsonl"),
    )
    parser.add_argument(
        "--eval-pool",
        type=Path,
        default=Path("data/issue_411/wrong_claims/eval_50.jsonl"),
    )
    parser.add_argument(
        "--slab-root",
        type=Path,
        default=Path("eval_results/issue_411"),
        help="Where per-source per-panel eval JSONs land.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/workspace/runs/issue_411"),
        help="Where per-cell adapter + (temporarily) merged dirs land.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    if args.smoke:
        sources = ["villain"]
    elif args.only_source:
        sources = [args.only_source]
    else:
        sources = args.sources

    log.info(
        "Dispatcher start — sources=%s seed=%d train_pool=%s eval_pool=%s slab=%s runs=%s",
        sources,
        args.seed,
        args.train_pool,
        args.eval_pool,
        args.slab_root,
        args.runs_root,
    )
    log.info("Smoke == sweep with one source: same _run_one_cell function path.")

    args.slab_root.mkdir(parents=True, exist_ok=True)
    args.runs_root.mkdir(parents=True, exist_ok=True)
    Path("/workspace/logs").mkdir(parents=True, exist_ok=True)

    per_cell: list[dict] = []
    plan_deviations: list[str] = []
    for source in sources:
        try:
            cell_summary = _run_one_cell(
                source=source,
                seed=args.seed,
                train_pool=args.train_pool,
                eval_pool=args.eval_pool,
                slab_root=args.slab_root,
                runs_root=args.runs_root,
            )
            per_cell.append(cell_summary)
        except Exception as e:
            # Loud fail per CLAUDE.md "fail fast" — re-raise after writing a
            # per-source failure sentinel so the orchestrator's poller picks
            # up the failure state.
            fail_path = Path(f"/workspace/logs/issue-411-{source}-FAILED.json")
            fail_path.parent.mkdir(parents=True, exist_ok=True)
            with open(fail_path, "w") as f:
                json.dump(
                    {
                        "source": source,
                        "phase": "cell_failed",
                        "exception_type": type(e).__name__,
                        "exception_msg": str(e),
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                    },
                    f,
                    indent=2,
                )
            log.exception("[%s] cell failed; wrote %s", source, fail_path)
            raise

    _write_final_sentinel(
        sources=sources,
        seed=args.seed,
        slab_root=args.slab_root,
        per_cell=per_cell,
        sources_requested=sources,
        plan_deviations=plan_deviations,
    )
    log.info("Dispatcher done. %d cells completed.", len(per_cell))
    return 0


if __name__ == "__main__":
    sys.exit(main())
