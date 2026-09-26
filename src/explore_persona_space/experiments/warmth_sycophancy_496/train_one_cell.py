"""Task #496 Phase 1 -- single-cell train+merge+eval+upload+cleanup.

One cell = one (arm, source, seed) triple. Pipeline:

1. Build the per-cell training pool (warmth arm: per-source contrastive; sycophancy
   arm: download #411's verbatim 700-row pool from HF).
2. ``train_lora`` -- LoRA SFT 700 rows, ~50 min on 1x H100.
3. ``merge_lora`` -- write merged Qwen+LoRA to ``<output_dir>/merged/``.
4. Upload merged adapter to HF Hub
   ``superkaiba1/explore-persona-space/issue496_<arm>_<source>_seed<seed>``.
5. Eval (vLLM batched, 24 panel x 50 claims x 10 rollouts) -- see ``eval_one_source``.
6. Write sentinel file at ``/workspace/logs/issue-496-<arm>-<source>-results.json``.
7. ``shutil.rmtree(output_dir / "merged")`` to free MooseFS quota before next cell.

All credentials and HF_TOKEN come from .env via ``dotenv.load_dotenv()``.

Per CLAUDE.md gotcha: ``train.sft.train_lora`` clobbers env CUDA_VISIBLE_DEVICES
with ``cfg.gpu_id`` (default 0). Pass ``gpu_id`` explicitly when running cells
in parallel across GPUs.

EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1 is honored by the trainer to skip WandB
Artifacts intermediate upload, since we do our own HF Hub push.
"""

from __future__ import annotations

import argparse
import gc
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

log = logging.getLogger("issue_496.train_one_cell")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _push_adapter_to_hub(merged_dir: Path, arm: str, source: str, seed: int) -> dict[str, str]:
    """Upload merged-model dir to HF Hub under
    ``superkaiba1/explore-persona-space``, path
    ``adapters/issue_496/<arm>_<source>_seed<seed>``.

    Returns {"repo_id", "path_in_repo", "commit_sha"}.
    """
    from huggingface_hub import HfApi

    path_in_repo = f"adapters/issue_496/{arm}_{source}_seed{seed}"
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=HF_MODEL_REPO, exist_ok=True, private=False)
    log.info("Uploading %s -> %s/%s ...", merged_dir, HF_MODEL_REPO, path_in_repo)
    commit_info = api.upload_folder(
        repo_id=HF_MODEL_REPO,
        folder_path=str(merged_dir),
        path_in_repo=path_in_repo,
        commit_message=f"issue #496 {arm} {source} seed {seed}",
        token=os.environ.get("HF_TOKEN"),
    )
    return {
        "repo_id": HF_MODEL_REPO,
        "path_in_repo": path_in_repo,
        "commit_sha": getattr(commit_info, "oid", None) or str(commit_info),
    }


def _write_sentinel(
    *,
    sentinel_dir: Path,
    arm: str,
    source: str,
    seed: int,
    phase: str,
    extra: dict[str, object] | None = None,
) -> Path:
    """Write the pod-side end-of-cell sentinel for ``poll_pipeline.py``."""
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    epoch = int(time.time())
    fname = f"issue-496-epm_results-{epoch}.json"
    payload: dict[str, object] = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "ts": datetime.now(UTC).isoformat(),
        "by": "issue_496.train_one_cell",
        "note": {
            "arm": arm,
            "source": source,
            "seed": seed,
            "phase": phase,
            "hostname": socket.gethostname(),
            "git_commit_sha": _git_sha(),
        },
    }
    if extra:
        payload["note"].update(extra)
    path = sentinel_dir / fname
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Wrote sentinel -> %s", path)
    return path


def run_one_cell(
    *,
    arm: str,
    source: str,
    seed: int,
    train_jsonl_path: Path,
    eval_pool_path: Path,
    output_dir: Path,
    eval_out_dir: Path,
    gpu_id: int = 0,
    sentinel_dir: Path = Path("/workspace/logs"),
    epochs: int = 3,
    lr: float = 1e-5,
    max_length: int = 1024,
    batch_size: int = 4,
    grad_accum: int = 4,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    panel_subset: list[str] | None = None,
    n_rollouts: int = 10,
    max_new_tokens: int = 512,
    do_upload: bool = True,
    do_eval: bool = True,
    rmtree_merged: bool = True,
    skip_train: bool = False,
) -> dict[str, object]:
    """Run one (arm, source, seed) cell end-to-end.

    Args mirror the dispatcher's CLI. Returns a dict summarizing each phase.
    """
    from explore_persona_space.train.sft import (
        TrainLoraConfig,
        merge_lora,
        train_lora,
    )

    log.info(
        "run_one_cell: arm=%s source=%s seed=%d gpu_id=%d output_dir=%s",
        arm,
        source,
        seed,
        gpu_id,
        output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "adapter"
    merged_dir = output_dir / "merged"
    summary: dict[str, object] = {
        "arm": arm,
        "source": source,
        "seed": seed,
        "gpu_id": gpu_id,
        "output_dir": str(output_dir),
        "phases": {},
        "started_utc": datetime.now(UTC).isoformat(),
    }

    if not skip_train:
        log.info("[phase=train] starting ...")
        t0 = time.time()
        cfg = TrainLoraConfig(
            gpu_id=gpu_id,
            epochs=epochs,
            lr=lr,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            batch_size=batch_size,
            grad_accum=grad_accum,
            max_length=max_length,
            seed=seed,
            run_name=f"issue496_{arm}_{source}_seed{seed}",
            report_to="none",
            hf_upload=False,  # We do our own upload after merge.
        )
        _out, loss = train_lora(
            base_model_path=BASE_MODEL,
            data_path=str(train_jsonl_path),
            output_dir=str(adapter_dir),
            cfg=cfg,
        )
        t_train = time.time() - t0
        summary["phases"]["train"] = {
            "wall_seconds": round(t_train, 1),
            "final_loss": float(loss),
            "adapter_dir": str(adapter_dir),
        }
        log.info("[phase=train] done in %.1fs loss=%.4f", t_train, loss)

        log.info("[phase=merge] starting ...")
        t1 = time.time()
        merged_path = merge_lora(
            base_model_path=BASE_MODEL,
            adapter_path=str(adapter_dir),
            output_dir=str(merged_dir),
            gpu_id=gpu_id,
        )
        gc.collect()
        t_merge = time.time() - t1
        summary["phases"]["merge"] = {
            "wall_seconds": round(t_merge, 1),
            "merged_dir": str(merged_path),
        }
        log.info("[phase=merge] done in %.1fs -> %s", t_merge, merged_path)

    if do_upload:
        log.info("[phase=upload] starting ...")
        t2 = time.time()
        upload_info = _push_adapter_to_hub(merged_dir, arm, source, seed)
        t_upload = time.time() - t2
        summary["phases"]["upload"] = {
            "wall_seconds": round(t_upload, 1),
            **upload_info,
        }
        log.info("[phase=upload] done in %.1fs -> %s", t_upload, upload_info)

    if do_eval:
        log.info("[phase=eval] starting ...")
        t3 = time.time()
        from explore_persona_space.experiments.warmth_sycophancy_496.eval_one_source import (
            eval_source,
        )

        eval_summary = eval_source(
            arm=arm,
            source=source,
            seed=seed,
            merged_model_path=merged_dir if merged_dir.exists() else None,
            hub_model_id=None if merged_dir.exists() else f"{HF_MODEL_REPO}",
            eval_pool_path=eval_pool_path,
            out_dir=eval_out_dir,
            panel_subset=panel_subset,
            n_rollouts=n_rollouts,
            max_new_tokens=max_new_tokens,
        )
        t_eval = time.time() - t3
        summary["phases"]["eval"] = {
            "wall_seconds": round(t_eval, 1),
            "eval_out_dir": str(eval_out_dir),
            "n_panel": eval_summary["n_panel"],
            "n_claims": eval_summary["n_claims"],
        }
        log.info("[phase=eval] done in %.1fs", t_eval)

    if rmtree_merged and merged_dir.exists():
        log.info("[phase=cleanup] rmtree %s ...", merged_dir)
        shutil.rmtree(merged_dir, ignore_errors=True)
        summary["phases"]["cleanup"] = {"rmtree": str(merged_dir)}

    summary["ended_utc"] = datetime.now(UTC).isoformat()
    log.info("[phase=done] arm=%s source=%s seed=%d", arm, source, seed)
    sent = _write_sentinel(
        sentinel_dir=sentinel_dir,
        arm=arm,
        source=source,
        seed=seed,
        phase="done",
        extra={"summary": summary["phases"]},
    )
    summary["sentinel_path"] = str(sent)
    return summary


def _main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", required=True, choices=["warmth", "sycophancy"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-jsonl", type=Path, required=True, help="Per-cell training pool JSONL."
    )
    parser.add_argument("--eval-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-out-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--sentinel-dir",
        type=Path,
        default=Path("/workspace/logs"),
        help="Where to write the end-of-cell poll_pipeline sentinel.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--panel-subset", nargs="*", default=None)
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--keep-merged", action="store_true")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Smoke aid: skip train+merge, go straight to eval (requires merged_dir exists).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=phase1] %(message)s")
    run_one_cell(
        arm=args.arm,
        source=args.source,
        seed=args.seed,
        train_jsonl_path=args.train_jsonl,
        eval_pool_path=args.eval_pool,
        output_dir=args.output_dir,
        eval_out_dir=args.eval_out_dir,
        gpu_id=args.gpu_id,
        sentinel_dir=args.sentinel_dir,
        epochs=args.epochs,
        lr=args.lr,
        max_length=args.max_length,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        panel_subset=args.panel_subset,
        n_rollouts=args.n_rollouts,
        max_new_tokens=args.max_new_tokens,
        do_upload=not args.no_upload,
        do_eval=not args.no_eval,
        rmtree_merged=not args.keep_merged,
        skip_train=args.skip_train,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
