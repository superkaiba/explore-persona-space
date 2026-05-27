"""Per-cell entrypoint for the task #397 Phase B sweep.

This module is the subprocess body invoked by
``scripts/dispatch_factor_screen_397.py::_launch_cell_subprocess``. ONE
process per (cell, source, seed) tuple. Each subprocess is pinned to a
single GPU via the ``--gpu-id`` flag (which both sets
``CUDA_VISIBLE_DEVICES`` in the subprocess env AND threads
``gpu_id=<N>`` into ``TrainLoraConfig`` so the
``train/sft.py:479`` clobber of CVD with ``cfg.gpu_id`` lands on the
right device — see the ``+gpu_id Hydra arg`` memory note).

Pipeline:

  1. ``prepare_cell_jsonl`` reads pools from ``--pool-dir`` (with C=1
     preflight + B=1 band assertion enabled by default), writes the
     per-cell training JSONL, returns the system_prompt_text.
  2. ``train_one_cell`` runs LoRA training + writes intermediate
     checkpoints + recipe-fix manifest. ``hf_upload=True`` lets
     ``train_lora`` push the final adapter to HF Hub.
  3. ``compute_logprob_panel`` runs per-checkpoint log-prob eval over
     the 480-context (24-persona x 20-question) train-matched panel
     using the peft 0.18.1 adapter-swap lifecycle.
  4. Final-checkpoint sampled eval writes ``metrics.json`` with the
     per-persona substring rate (M1 source-rate check).
  5. ``--verify-hf-upload`` (default on) probes HF Hub to confirm the
     adapter landed before per-cell cleanup runs. On verify-failure the
     cleanup is SKIPPED and the subprocess exits non-zero so the
     dispatcher logs a per-cell failure marker.
  6. On verify-PASS, per-cell cleanup removes ``merged/`` + each
     ``checkpoint-*/`` directory (plan §11 disk-quota discipline).

The subprocess return code maps to per-cell status:
  - 0   = ok (train + eval + upload-verify + cleanup all succeeded)
  - 1   = train or eval crashed (exception caught and logged; cleanup skipped)
  - 2   = upload-verify FAILED (local weights preserved for manual recovery)
  - 3   = unexpected (CLI parse error, missing pool, etc.)

The dispatcher logs the cell's failure but DOES NOT kill the sweep on
non-zero returns — single-cell failures are isolated.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

log = logging.getLogger("run_one_cell")


def _setup_logging(log_level: str, log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_one_cell",
        description=(
            "Per-cell entrypoint for task #397 Phase B sweep. "
            "ONE process per (cell, source, seed); one GPU per process."
        ),
    )
    p.add_argument("--cell", type=str, required=True, help="5-char cell key, e.g. 10012")
    p.add_argument("--source", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpu-id", type=int, required=True)
    p.add_argument("--pool-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--marker-token", type=str, default="※")
    p.add_argument("--save-every-n-steps", type=int, default=25)
    p.add_argument("--pos-per-source", type=int, default=400)
    p.add_argument("--neg-per-source", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.10)
    p.add_argument(
        "--verify-hf-upload",
        action="store_true",
        default=True,
        help="Probe HF Hub for the adapter before per-cell cleanup (default ON).",
    )
    p.add_argument(
        "--skip-hf-upload-verify",
        action="store_true",
        help="Bypass the HF-upload verification gate (DANGEROUS — preserves local weights).",
    )
    p.add_argument(
        "--skip-cleanup",
        action="store_true",
        help=(
            "Skip per-cell cleanup of merged/ + checkpoint-*/ (preserves all local "
            "weights; used during smoke / debugging)."
        ),
    )
    p.add_argument("--log-level", type=str, default="INFO")
    return p


# ---------------------------------------------------------------------------
# HF Hub upload verification + per-cell cleanup
# ---------------------------------------------------------------------------


def verify_adapter_on_hf_hub(*, hf_path_in_repo: str, repo_id: str) -> bool:
    """Probe HF Hub to confirm an adapter directory exists under ``hf_path_in_repo``.

    Returns True if at least one ``adapter_*`` file (e.g. ``adapter_model.safetensors``,
    ``adapter_config.json``) is present at the path. Returns False on missing path
    OR transient Hub failure — caller treats False as "do not delete local weights".

    Per CLAUDE.md upload policy: "Models MUST upload to HF model repo before local
    deletion. Never delete unuploaded." This helper is the gate that enforces it.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.error("huggingface_hub not importable; cannot verify HF upload")
        return False

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as e:
        log.error("HF Hub list_repo_files failed (%s); cannot verify upload", e)
        return False

    prefix = hf_path_in_repo.rstrip("/") + "/"
    found = [f for f in files if f.startswith(prefix) and ("adapter_" in f.rsplit("/", 1)[-1])]
    if not found:
        log.warning(
            "HF Hub verification: NO adapter files under %s/%s — refusing to delete locally",
            repo_id,
            hf_path_in_repo,
        )
        return False
    log.info(
        "HF Hub verification PASS: %d adapter file(s) at %s/%s",
        len(found),
        repo_id,
        hf_path_in_repo,
    )
    return True


def cleanup_cell_local_weights(cell_output_dir: Path) -> dict[str, int]:
    """Remove merged/ + checkpoint-*/ directories after upload-verify PASS.

    Plan v4 §11 disk-quota discipline: peak disk per cell ~3 GB
    (intermediate checkpoint dirs) post-Round-6. The merge step that
    drove the ~14 GB footprint has been removed; vLLM ``--enable-lora``
    consumes the adapter directly without merging. Keep the per-cell
    ``metrics.json`` + ``logprob_*.json`` + ``prepared_dataset.json`` +
    ``run.log`` — they're small text + needed for diagnosis.

    Returns ``{"merged_removed": 0|1, "checkpoints_removed": N}`` for
    bookkeeping. ``merged_removed`` is retained for backward-compat with
    pre-Round-6 cell dirs that may carry a stale ``merged/`` from an
    older training run; new Round-6 runs always report 0 there.
    """
    removed = {"merged_removed": 0, "checkpoints_removed": 0}
    merged_dir = cell_output_dir / "merged"
    if merged_dir.is_dir():
        shutil.rmtree(merged_dir)
        removed["merged_removed"] = 1
        log.info("Cleanup: removed %s", merged_dir)
    adapter_dir = cell_output_dir / "adapter"
    if adapter_dir.is_dir():
        for ck in sorted(adapter_dir.glob("checkpoint-*")):
            if ck.is_dir():
                shutil.rmtree(ck)
                removed["checkpoints_removed"] += 1
        log.info(
            "Cleanup: removed %d checkpoint dir(s) under %s",
            removed["checkpoints_removed"],
            adapter_dir,
        )
    return removed


# ---------------------------------------------------------------------------
# Per-cell pipeline (train → eval → sampled eval → upload-verify → cleanup)
# ---------------------------------------------------------------------------


def run_cell(args: argparse.Namespace) -> int:
    """Execute the full per-cell pipeline. Returns OS exit code."""
    # Set CUDA_VISIBLE_DEVICES BEFORE any torch / vLLM / HF Transformers import
    # so the entire process sees only the assigned GPU. train_one_cell will
    # also set this (via TrainLoraConfig.gpu_id → sft.py:479) — redundant but
    # harmless. The redundancy is per the +gpu_id memory note: env CVD alone
    # is insufficient because sft.py overwrites it with cfg.gpu_id (default 0).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Heavy imports deferred so CLI / arg-validation paths can be tested
    # without pulling in torch / TRL / vLLM.
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
        EVAL_QUESTIONS_20,
    )
    from explore_persona_space.experiments.factor_screen_397.cells import Cell
    from explore_persona_space.experiments.factor_screen_397.data_prep import (
        prepare_cell_jsonl,
    )
    from explore_persona_space.experiments.factor_screen_397.eval_panel import (
        FINAL_CHECKPOINT_MARKER_VARIANTS,
        build_train_matched_persona_panel,
        compute_logprob_panel,
        read_prepared_dataset_manifest,
    )
    from explore_persona_space.experiments.factor_screen_397.training import (
        BASE_MODEL,
        train_one_cell,
    )

    cell = Cell.from_key(args.cell)
    cell_output_dir = Path(args.output_dir)
    cell_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = cell_output_dir / "run.log"
    _setup_logging(args.log_level, log_file)

    log.info(
        "run_one_cell starting: cell=%s source=%s seed=%d e=%d gpu=%d -> %s",
        cell.key,
        args.source,
        args.seed,
        cell.e,
        args.gpu_id,
        cell_output_dir,
    )

    # Load tokenizer once for the C=1 preflight (Round 5 hardening).
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- (1) Data-prep -----
    data_path = cell_output_dir / "prepared_train.jsonl"
    log.info("Preparing cell JSONL from pool dir %s -> %s", args.pool_dir, data_path)
    prep_result = prepare_cell_jsonl(
        cell=cell,
        source=args.source,
        pool_dir=args.pool_dir,
        output_path=data_path,
        marker_text=args.marker_token,
        pos_per_source=args.pos_per_source,
        neg_per_source=args.neg_per_source,
        seed=args.seed,
        tokenizer=tokenizer,
        enforce_c_preflight=True,
        enforce_b1_band=True,
    )
    system_prompt_text = prep_result["system_prompt_text"]
    log.info(
        "Data-prep complete: %d pos + %d neg = %d rows (data_policy=%s)",
        prep_result["num_positive"],
        prep_result["num_negative"],
        prep_result["num_total"],
        prep_result["data_policy"],
    )

    # ----- (2) Train -----
    # Round 10: hf_upload=False. The TRL inline-upload fence (sft.py:667)
    # is wrapped in `except Exception` — any upload failure is silently
    # swallowed (logger.warning only), leaving train_lora returning
    # success while the adapter is NOT on Hub. Sweep cells 00001/00002/
    # 00011 all hit this on the first launch → verify_adapter_on_hf_hub
    # returned False → rc=2 → ~321 GB of local weights would have blown
    # past the MooseFS quota.
    #
    # Round 10 fix moves upload to step (5) below where it's explicit +
    # fail-loud. Setting hf_upload=False here avoids the double-upload
    # (HF Hub upserts under the same path, but it's wasteful) AND keeps
    # the upload-failure surface in run_one_cell's hand (where rc maps
    # to per-cell failure cleanly).
    train_start = time.time()
    outcome = train_one_cell(
        cell=cell,
        seed=args.seed,
        source=args.source,
        data_path=data_path,
        cell_output_dir=cell_output_dir,
        marker_text=args.marker_token,
        save_every_n_steps=args.save_every_n_steps,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        gpu_id=args.gpu_id,  # threaded to TrainLoraConfig.gpu_id (clobber-safe)
        hf_upload=False,  # Round 10 — explicit upload in step (5)
        system_prompt_text=system_prompt_text,
    )
    train_minutes = (time.time() - train_start) / 60.0
    log.info("Training complete: %.2f min, loss=%.4f", train_minutes, outcome.loss)

    # ----- (3) Train-matched panel + per-checkpoint log-prob eval -----
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    adapter_dir = cell_output_dir / "adapter"
    checkpoint_dirs = sorted(
        (str(c) for c in adapter_dir.glob("checkpoint-*") if c.is_dir()),
        key=lambda p: int(p.rsplit("-", 1)[-1]),
    )
    if not checkpoint_dirs:
        log.error("No intermediate checkpoint dirs under %s", adapter_dir)
        return 1

    manifest = read_prepared_dataset_manifest(cell_output_dir)
    panel, overrides = build_train_matched_persona_panel(
        canonical_panel=EVAL_PERSONAS_24,
        source=args.source,
        manifest=manifest,
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
        device_map="auto",
    )
    peft_model = PeftModel.from_pretrained(base, checkpoint_dirs[0], adapter_name="ck0")

    questions = list(EVAL_QUESTIONS_20)
    log.info(
        "Log-prob eval: %d personas x %d questions = %d contexts x %d checkpoints",
        len(panel),
        len(questions),
        len(panel) * len(questions),
        len(checkpoint_dirs),
    )
    eval_start = time.time()
    logprob_result = compute_logprob_panel(
        base_model=peft_model,
        tokenizer=tokenizer,
        checkpoint_dirs=checkpoint_dirs,
        personas=panel,
        questions=questions,
        system_prompt_overrides=overrides,  # SR1 wiring
        marker_texts=FINAL_CHECKPOINT_MARKER_VARIANTS,
        batch_size=8,
        device="cuda:0",  # CVD pinned the process to one GPU
    )
    eval_minutes = (time.time() - eval_start) / 60.0
    log.info("Log-prob eval complete: %.2f min", eval_minutes)
    logprob_path = cell_output_dir / "logprob_panel.json"
    logprob_path.write_text(json.dumps(logprob_result, indent=2), encoding="utf-8")
    log.info("Wrote %s", logprob_path)

    # Round 8 fix — aggressive HF teardown before vLLM init.
    #
    # First-launch crash (smoke cell 10010): HF Transformers held ~36 GB on
    # GPU 0 from log-prob eval; round-6's del + gc.collect + empty_cache
    # was insufficient — PyTorch caching-allocator blocks persist; vLLM
    # tried to grab 0.6 * 79 GB = 47.5 GB; only 43.3 GB free → instant
    # ValueError. Defense-in-depth pattern:
    #
    #   1. del every named ref to the HF model / peft model / tokenizer
    #      (any list / dict / class attr that holds them blocks GC).
    #   2. gc.collect() to clear Python refs.
    #   3. torch.cuda.empty_cache() to release the PyTorch caching
    #      allocator blocks.
    #   4. torch.cuda.synchronize() to ensure pending CUDA ops finish
    #      BEFORE we read mem-info or hand control to vLLM.
    #   5. Log pre/post free-memory so the next OOM is debuggable.
    #
    # Even with all 4 steps the residue can still be non-zero (PyTorch
    # holds some allocator overhead). The vLLM-side defense is
    # gpu_memory_utilization=0.45 (Round 8 Fix 2), which leaves the
    # ~36 GB headroom the residue needs.
    import gc

    import torch

    free_before_gb = torch.cuda.mem_get_info()[0] / (1024**3) if torch.cuda.is_available() else -1.0

    # Step 1: drop every Python ref to the HF stack. compute_logprob_panel
    # returned plain dicts of floats, so logprob_result holds no GPU refs.
    # peft_model and base hold the GPU weights; tokenizer holds none (CPU
    # only) but we del it for completeness.
    del peft_model, base
    del tokenizer

    # Step 2 + 3 + 4: GC + cache release + sync.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_after_gb = torch.cuda.mem_get_info()[0] / (1024**3)
        log.info(
            "HF teardown before vLLM: free GPU memory %.2f GB → %.2f GB "
            "(residue %.2f GB; vLLM will request 0.45 * total)",
            free_before_gb,
            free_after_gb,
            free_before_gb - free_after_gb if free_after_gb > free_before_gb else 0.0,
        )

    # ----- (4) Final-checkpoint sampled eval via vLLM --enable-lora -----
    # Round 6: no merge step. vLLM loads BASE model once with
    # enable_lora=True, then LoRARequest hands the adapter at inference
    # time. Eliminates the ~14 GB merged-dir per cell.
    from explore_persona_space.experiments.factor_screen_397.eval_panel import (
        DEFAULT_NUM_COMPLETIONS,
        generate_completions_with_lora,
        score_markers_threaded,
    )

    log.info(
        "Sampled eval starting: base=%s lora_path=%s (vLLM --enable-lora; NO merge)",
        BASE_MODEL,
        outcome.adapter_path,
    )
    completions = generate_completions_with_lora(
        base_model_path=BASE_MODEL,
        lora_path=outcome.adapter_path,
        personas=dict(panel),
        questions=questions,
        system_prompt_overrides=overrides,  # SR1 wiring
        seed=args.seed,
    )
    persona_scores = score_markers_threaded(completions, marker=args.marker_token)

    source_rate = persona_scores.get(args.source, {}).get("substring_rate")
    metrics_payload = {
        "marker": args.marker_token,
        "cell_key": cell.key,
        "source": args.source,
        "seed": args.seed,
        "e": cell.e,
        "train_wall_minutes": train_minutes,
        "logprob_eval_wall_minutes": eval_minutes,
        "panel_size": len(panel),
        "questions": len(questions),
        "num_completions": DEFAULT_NUM_COMPLETIONS,
        "personas": persona_scores,
        "source_substring_rate": source_rate,
        "vllm_lora_mode": True,  # round 6 marker — no merged dir consumed
    }
    metrics_path = cell_output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    log.info(
        "Sampled eval complete: wrote %s (source_rate=%s)",
        metrics_path,
        f"{source_rate:.3f}" if source_rate is not None else "None",
    )

    # ----- (5) HF Hub upload (Round 10 fix — was implicit before) -----
    # Round 9 + earlier relied on TRL's inline-upload fence inside
    # train_lora to push the adapter to HF Hub. That path is wrapped in
    # `except Exception` in sft.py:681 — any upload failure (transient
    # network blip, rate limit, missing repo perms) is logged-and-
    # swallowed; train_one_cell returns success; then verify_adapter_on_hf_hub
    # finds nothing on Hub → rc=2 → local weights preserved → MooseFS
    # quota blown on the third such cell. Sweep crashed at cell 3 of 7
    # with 3 rc=2 failures.
    #
    # Fix: explicit upload here, BEFORE the verify gate. If upload raises,
    # we exit non-zero immediately (fail-fast) instead of letting verify
    # surface the absence. If upload silently succeeds-but-doesn't-land
    # (the original failure mode), verify catches it as the safety net.
    run_name = f"i397_cell_{cell.key}_source_{args.source}_seed{args.seed}"
    hf_path_in_repo = f"adapters/issue_397/{run_name}"
    from explore_persona_space.orchestrate.hub import upload_model

    log.info(
        "Uploading adapter to HF Hub: %s -> superkaiba1/explore-persona-space/%s",
        outcome.adapter_path,
        hf_path_in_repo,
    )
    upload_start = time.time()
    hub_path = upload_model(
        outcome.adapter_path,
        repo_id="superkaiba1/explore-persona-space",
        path_in_repo=hf_path_in_repo,
    )
    upload_minutes = (time.time() - upload_start) / 60.0
    if not hub_path:
        # upload_model returns "" on failure (per its docstring); raise so
        # the per-cell rc reflects the failure and cleanup does NOT run.
        log.error(
            "HF upload returned empty path for %s — failing cell (local weights preserved at %s)",
            hf_path_in_repo,
            cell_output_dir,
        )
        return 2
    log.info(
        "HF upload complete: %s (%.2f min)",
        hub_path,
        upload_minutes,
    )

    # ----- (6) HF Hub upload verification gate (safety net) -----
    if args.skip_hf_upload_verify:
        log.warning(
            "--skip-hf-upload-verify is set; cleanup will run WITHOUT confirming "
            "the adapter landed on HF Hub. This is DANGEROUS — only use for debug."
        )
        upload_verified = True
    elif args.verify_hf_upload:
        upload_verified = verify_adapter_on_hf_hub(
            hf_path_in_repo=hf_path_in_repo,
            repo_id="superkaiba1/explore-persona-space",
        )
    else:
        upload_verified = True

    if not upload_verified:
        log.error(
            "HF upload verification FAILED for %s — preserving local weights "
            "at %s for manual recovery. Cell exits with rc=2.",
            hf_path_in_repo,
            cell_output_dir,
        )
        return 2

    # ----- (6) Per-cell cleanup -----
    if args.skip_cleanup:
        log.info("--skip-cleanup set; preserving merged/ + checkpoint-*/")
    else:
        removed = cleanup_cell_local_weights(cell_output_dir)
        log.info("Per-cell cleanup: %s", removed)

    log.info(
        "run_one_cell complete: cell=%s source=%s seed=%d → rc=0", cell.key, args.source, args.seed
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return run_cell(args)
    except Exception:
        # Top-level safety net: log full traceback + return rc=1 so the
        # dispatcher can mark the cell as failed without taking down the
        # sweep. The traceback lands in the per-cell run.log via the
        # FileHandler installed inside run_cell.
        log.exception(
            "run_one_cell crashed for cell=%s source=%s seed=%d", args.cell, args.source, args.seed
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
