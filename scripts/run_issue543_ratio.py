#!/usr/bin/env python3
"""Issue #543 dispatcher — ratio-lever marker install + benign-SFT survival.

4 arms (r50/r25/r10/r05) x 3 seeds x 2 phases on one 4x H100 pod. SMOKE IS
SWEEP WITH ONE CELL (plan §4.5): the smoke stage runs the IDENTICAL per-cell
pipeline (`--cell` -> phase1 train -> smoke eval -> phase2 train -> smoke
eval) on (r50, seed 42) through the same subprocess/env/logging/sentinel
path, gates on plan §7 (stop predicate within 1500 steps + dev emission
>= 48/50), and only then fans the remaining 11 cells out over 4 GPUs
(3 cells per GPU, sequential per GPU; GPU pinning via TrainLoraConfig.gpu_id
/ eval --gpu — never bare env CUDA_VISIBLE_DEVICES).

Phase 1 (install, marker-only loss, band-stop matching — plan §4.2):
  - b-hat pre-pass: base-model mean log P(marker) on the frozen 32-row
    trigger probe, ONE shared constant for all 12 cells (sanity [-23, -15]).
  - stop: trained mean log P(marker) in [-0.45, -0.05] absolute (delta band
    [low - b_hat, high - b_hat]) AND slot-argmax >= 31/32, probes every 5
    steps, min 20; overshoot-stop + rolling-checkpoint nearest-band
    selection; 16-epoch cap breach = pre-registered finding.
  - post-stop on-policy dev check (>= 48/50 emissions; one band+0.10 retry).
Phase 2 (erasure, plan §4.3): 1 epoch good_medical_advice_6k, assistant-CE,
  lr 1e-4 cosine, continue-adapter, band-stop OFF, 4 trajectory-only
  callbacks every 5 steps.

Pod-side contract: per-phase sentinels /workspace/logs/issue-543-*.json
(poll_pipeline schema v1) + ``[phase=...]`` milestones; the driver emits the
terminal ``[phase=done]`` ONLY on full success. The pod NEVER shells out to
scripts/task.py.

Usage (pod):
    nohup uv run python scripts/run_issue543_ratio.py --driver &
    uv run python scripts/run_issue543_ratio.py --cell --arm r50 --seed 42 --gpu 0
    uv run python scripts/run_issue543_ratio.py --phase phase1 --arm r50 --seed 42 --gpu 0
Local (no GPU): --plan-only walks the driver logic + sentinel writer.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="run_issue543_ratio")

import os  # noqa: E402

from _issue543_common import (  # noqa: E402
    ARMS,
    BAND_RETRY_SHIFT_NATS,
    BANK_CLASSES,
    BANK_DIR,
    BASE_MODEL,
    BHAT_PATH,
    BHAT_SANITY_RANGE,
    DATA_SEED,
    EOS_TOKEN_ID,
    EVAL_RESULTS_DIR,
    EXPECTED_MARKER_ID,
    HUB_MODEL_REPO,
    MARKER_TEXT,
    MIX_MANIFEST_PATH,
    MIXES_DIR,
    N_PROBE_ROWS,
    N_TRAIN_QUESTIONS,
    PHASE1_BAND_EVAL_EVERY,
    PHASE1_BAND_MIN_STEPS,
    PHASE1_EPOCHS_CAP,
    PHASE1_GRAD_ACCUM,
    PHASE1_LORA_ALPHA,
    PHASE1_LORA_DROPOUT,
    PHASE1_LORA_R,
    PHASE1_LORA_TARGETS,
    PHASE1_LR,
    PHASE1_LR_SCHEDULER,
    PHASE1_MAX_LENGTH,
    PHASE1_MIN_ARGMAX_RATE,
    PHASE1_PER_DEVICE_BS,
    PHASE1_SAVE_STEPS,
    PHASE1_SAVE_TOTAL_LIMIT,
    PHASE1_WARMUP_RATIO,
    PHASE2_DATASET_HF_PATH,
    PHASE2_DATASET_REL,
    PHASE2_EPOCHS,
    PHASE2_EXPECTED_ROWS,
    PHASE2_GRAD_ACCUM,
    PHASE2_LR,
    PHASE2_LR_SCHEDULER,
    PHASE2_MAX_LENGTH,
    PHASE2_PER_DEVICE_BS,
    PHASE2_TRAJECTORY_EVERY,
    PHASE2_WARMUP_RATIO,
    PHASES,
    PROBE_LOG_PREFIXES,
    PROBES_DIR,
    PROJECT_ROOT,
    SEEDS,
    STOP_TARGET_LOGP_HIGH,
    STOP_TARGET_LOGP_LOW,
    TOTAL_ROWS,
    WANDB_PROJECT,
    adapter_subfolder,
    cell_slug,
    marker_preflight,
    output_root,
    phase_log,
    repro_metadata,
    run_name,
    sentinel_dir,
    write_sentinel,
)

log = logging.getLogger("run_issue543_ratio")

N_GPUS = 4
SMOKE_GATE_MAX_STEPS = 1500  # plan §7: smoke cell must stop within 4 epochs
SMOKE_CELL = ("r50", 42)


# ── Small helpers ───────────────────────────────────────────────────────────


def _cell_dir(arm: str, seed: int) -> Path:
    return EVAL_RESULTS_DIR / arm / f"seed{seed}"


def _phase1_result_path(arm: str, seed: int) -> Path:
    return _cell_dir(arm, seed) / "phase1_result.json"


def _phase2_result_path(arm: str, seed: int) -> Path:
    return _cell_dir(arm, seed) / "phase2_result.json"


def _stop_record_path(arm: str, seed: int) -> Path:
    return _cell_dir(arm, seed) / "phase1_stop_record.json"


def _run_child(cmd: list[str], log_path: Path, *, label: str) -> None:
    """Run a child process with EXPLICIT env passthrough; fail loud with tail."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("[%s] spawning: %s (log=%s)", label, " ".join(cmd), log_path)
    env = {**os.environ}
    with log_path.open("ab") as logf:
        proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        tail = ""
        try:
            with log_path.open("rb") as f:
                f.seek(max(0, log_path.stat().st_size - 4096))
                tail = f.read().decode("utf-8", errors="replace")
        except OSError:
            pass
        raise RuntimeError(f"[{label}] child failed (rc={proc.returncode}); log tail:\n{tail}")


def _self_cmd(*args: str) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), *args]


def _eval_cmd(*args: str) -> list[str]:
    return [sys.executable, str(_SCRIPTS_DIR / "eval_issue543.py"), *args]


def _assert_credentials() -> None:
    """uv run does NOT auto-load .env; _bootstrap did — verify it took."""
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN missing from env — .env not loaded; aborting before launch.")


# ── Probe batches + trajectory callbacks ────────────────────────────────────


def _build_probe_callback(
    probe_file: str,
    tokenizer,
    *,
    band_low_delta: float,
    band_high_delta: float,
    stop_enabled: bool,
    eval_every: int,
    dump_path: Path,
    stop_record_path: Path | None = None,
    min_argmax_rate: float | None = None,
    overshoot_stop: bool = False,
    min_steps: int = 0,
):
    """Construct one MarkerBandStopCallback over a frozen probe JSONL."""
    from explore_persona_space.eval.callbacks import MarkerBandStopCallback
    from explore_persona_space.train.sft import build_source_probe_from_data

    path = PROBES_DIR / probe_file
    if not path.exists():
        raise FileNotFoundError(f"Probe file missing: {path}. Run build_issue543_mixes.py first.")
    input_ids, attention_mask, marker_positions, n_rows = build_source_probe_from_data(
        path,
        tokenizer,
        [EXPECTED_MARKER_ID],
        max_rows=64,  # probe files hold exactly the frozen rows; take all
        max_length=max(PHASE1_MAX_LENGTH, 2048),
    )
    if n_rows == 0:
        raise RuntimeError(f"Probe {probe_file} yielded 0 usable rows — build is broken.")
    return MarkerBandStopCallback(
        marker_token_ids=[EXPECTED_MARKER_ID],
        probe_input_ids=input_ids,
        probe_marker_positions=marker_positions,
        probe_attention_mask=attention_mask,
        low_nats=band_low_delta,
        high_nats=band_high_delta,
        eval_every_steps=eval_every,
        min_steps=min_steps,
        log_prefix=PROBE_LOG_PREFIXES[probe_file],
        eos_token_id=EOS_TOKEN_ID,
        min_argmax_rate=min_argmax_rate,
        stop_enabled=stop_enabled,
        overshoot_stop=overshoot_stop,
        dump_jsonl_path=str(dump_path),
        stop_record_path=str(stop_record_path) if stop_record_path else None,
    )


def _bystander_callbacks(
    tokenizer, *, cell: Path, phase: str, eval_every: int, include_trigger: bool
) -> list:
    """Trajectory-only callbacks (stop_enabled=False) for the frozen probes."""
    probes = list(PROBE_LOG_PREFIXES)
    if not include_trigger:
        probes = [p for p in probes if p != "probe_trigger.jsonl"]
    return [
        _build_probe_callback(
            p,
            tokenizer,
            band_low_delta=0.0,
            band_high_delta=1.0,  # unused (stop disabled); must satisfy low < high
            stop_enabled=False,
            eval_every=eval_every,
            dump_path=cell
            / f"{phase}_trajectory_{PROBE_LOG_PREFIXES[p].removeprefix('marker_')}.jsonl",
        )
        for p in probes
    ]


# ── b-hat pre-pass (one shared constant for all 12 cells) ───────────────────


def measure_bhat(gpu: int) -> float:
    """Base-model mean log P(marker) on the frozen trigger probe.

    Reuses the EXACT callback read path (same fused tokenization, same slot)
    so the band the stop decision uses is measured the way it will be read.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.train.sft import _pick_attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    cb = _build_probe_callback(
        "probe_trigger.jsonl",
        tokenizer,
        band_low_delta=0.0,
        band_high_delta=1.0,
        stop_enabled=False,
        eval_every=1,
        dump_path=Path("/tmp/issue543_bhat_probe.jsonl"),
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation=_pick_attn_implementation(),
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    stats = cb._compute_marker_slot_stats(model)
    bhat = float(stats["logp"].mean().item())
    lo, hi = BHAT_SANITY_RANGE
    if not (lo <= bhat <= hi):
        raise RuntimeError(
            f"b-hat sanity FAIL: measured {bhat:.3f} outside [{lo}, {hi}] — "
            "probe construction or tokenizer drift; refusing to set the band."
        )
    BHAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BHAT_PATH.write_text(
        json.dumps(
            {
                **repro_metadata(),
                "bhat_mean_logp": bhat,
                "n_probe_rows": int(stats["logp"].shape[0]),
                "per_row_logp": [float(v) for v in stats["logp"].tolist()],
            },
            indent=2,
        )
    )
    log.info("b-hat = %.4f nat (-> %s)", bhat, BHAT_PATH)
    del model
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return bhat


def _load_bhat() -> float:
    if not BHAT_PATH.exists():
        raise FileNotFoundError(
            f"{BHAT_PATH} missing — run `run_issue543_ratio.py --measure-bhat --gpu 0` first "
            "(the driver does this automatically)."
        )
    return float(json.loads(BHAT_PATH.read_text())["bhat_mean_logp"])


# ── Phase 1 ─────────────────────────────────────────────────────────────────


def _free_training_residue() -> None:
    """Release dead training refs + CUDA cache before a vLLM child spawns.

    ``train_lora`` leaves ~15.8 GiB resident in this process after returning
    (model/trainer reference-cycle garbage held by the caching allocator); a
    dev-check / eval child's vLLM init then fails its free-memory startup
    check (2026-06-10 smoke-cell incident: 63.4/79.2 GiB free < 0.9 util).
    """
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def _phase1_train_once(
    args: argparse.Namespace,
    *,
    bhat: float,
    band_low_abs: float,
    band_high_abs: float,
    out_dir: Path,
    existing_adapter: str | None,
    run_suffix: str,
    min_steps: int,
) -> tuple[str, dict]:
    """One train_lora invocation of the Phase-1 install recipe."""
    from transformers import AutoTokenizer

    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    arm, seed = args.arm, args.seed
    cell = _cell_dir(arm, seed)
    cell.mkdir(parents=True, exist_ok=True)
    data_path = MIXES_DIR / arm / "train.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Mix missing: {data_path}. Run build_issue543_mixes.py first.")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    callbacks = _bystander_callbacks(
        tokenizer,
        cell=cell,
        phase="phase1",
        eval_every=PHASE1_BAND_EVAL_EVERY,
        include_trigger=False,
    )

    stop_record_tmp = out_dir / "callback_stop_record.json"
    cfg = TrainLoraConfig(
        gpu_id=args.gpu,
        epochs=PHASE1_EPOCHS_CAP,
        lr=PHASE1_LR,
        lr_scheduler_type=PHASE1_LR_SCHEDULER,
        warmup_ratio=PHASE1_WARMUP_RATIO,
        lora_r=PHASE1_LORA_R,
        lora_alpha=PHASE1_LORA_ALPHA,
        lora_dropout=PHASE1_LORA_DROPOUT,
        lora_targets=list(PHASE1_LORA_TARGETS),
        batch_size=PHASE1_PER_DEVICE_BS,
        grad_accum=PHASE1_GRAD_ACCUM,
        max_length=PHASE1_MAX_LENGTH,
        seed=seed,
        run_name=run_name(arm, seed, "phase1") + run_suffix,
        report_to="wandb",
        save_strategy="steps",
        save_steps=PHASE1_SAVE_STEPS,
        save_total_limit=PHASE1_SAVE_TOTAL_LIMIT,
        logging_steps=5,
        marker_only_loss=True,
        marker_text=MARKER_TEXT,
        marker_tail_tokens=0,
        marker_suppress_at_post_response_slot=True,
        marker_im_end_token_id=EOS_TOKEN_ID,
        marker_band_stop=True,
        marker_band_low_nats=band_low_abs - bhat,
        marker_band_high_nats=band_high_abs - bhat,
        marker_band_eval_every_steps=PHASE1_BAND_EVAL_EVERY,
        marker_band_min_steps=min_steps,
        marker_band_min_argmax_rate=PHASE1_MIN_ARGMAX_RATE,
        marker_band_overshoot_stop=True,
        marker_band_dump_jsonl_path=str(cell / "phase1_trajectory_trigger.jsonl"),
        marker_band_log_prefix="marker_trigger",
        marker_band_stop_record_path=str(stop_record_tmp),
        hf_upload=True,
        hf_repo=HUB_MODEL_REPO,
        hf_path_in_repo=f"adapters/{adapter_subfolder(arm, seed, 'phase1')}",
        existing_adapter_path=existing_adapter,
    )

    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    adapter_path, train_loss = train_lora(
        base_model_path=BASE_MODEL,
        data_path=str(data_path),
        output_dir=str(out_dir / "adapter"),
        cfg=cfg,
        callbacks=callbacks,  # 3 bystander trajectory-only probes (plan §4.2)
    )
    _free_training_residue()
    callback_record = json.loads(stop_record_tmp.read_text()) if stop_record_tmp.exists() else None
    if callback_record is None:
        raise RuntimeError(
            f"Band-stop callback wrote no stop record at {stop_record_tmp} — "
            "the callback did not attach (0 marker rows?); the matching control is broken."
        )
    return adapter_path, {"train_loss": train_loss, **callback_record}


def _select_nearest_band_checkpoint(
    out_dir: Path, *, band_low_abs: float, band_high_abs: float, gpu: int
) -> tuple[Path, dict]:
    """Overshoot recovery (plan §4.2.4): re-probe the rolling checkpoints and
    pick the one whose trained mean log P(marker) is nearest the band midpoint."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.train.sft import _pick_attn_implementation

    ckpts = sorted(
        (p for p in (out_dir / "adapter").glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not ckpts:
        raise RuntimeError(f"Overshoot recovery: no rolling checkpoints under {out_dir}/adapter")
    midpoint = (band_low_abs + band_high_abs) / 2.0

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    cb = _build_probe_callback(
        "probe_trigger.jsonl",
        tokenizer,
        band_low_delta=0.0,
        band_high_delta=1.0,
        stop_enabled=False,
        eval_every=1,
        dump_path=out_dir / "overshoot_probe_scratch.jsonl",
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation=_pick_attn_implementation(),
        token=os.environ.get("HF_TOKEN"),
    )
    model = PeftModel.from_pretrained(base, str(ckpts[0]), adapter_name="ckpt_0")
    for i, ck in enumerate(ckpts[1:], start=1):
        model.load_adapter(str(ck), adapter_name=f"ckpt_{i}")
    model.eval()
    readings: list[dict] = []
    for i, ck in enumerate(ckpts):
        model.set_adapter(f"ckpt_{i}")
        logp = float(cb._compute_marker_slot_stats(model)["logp"].mean().item())
        readings.append(
            {"checkpoint": str(ck), "step": int(ck.name.split("-")[-1]), "mean_logp": logp}
        )
        log.info("Overshoot re-probe %s: mean logp %.4f (midpoint %.3f)", ck.name, logp, midpoint)
    del model, base
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    best = min(readings, key=lambda r: abs(r["mean_logp"] - midpoint))
    selected_dir = out_dir / "adapter_selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    src = Path(best["checkpoint"])
    copied = []
    for fname in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        if (src / fname).exists():
            shutil.copy2(src / fname, selected_dir / fname)
            copied.append(fname)
    if "adapter_config.json" not in copied or not any(
        f.startswith("adapter_model") for f in copied
    ):
        raise RuntimeError(f"Selected checkpoint {src} lacks adapter files (found {copied}).")
    selection = {"readings": readings, "selected": best, "selected_dir": str(selected_dir)}
    log.info(
        "Overshoot selection: step %d (logp %.4f) -> %s",
        best["step"],
        best["mean_logp"],
        selected_dir,
    )
    return selected_dir, selection


def _run_dev_check(args: argparse.Namespace, adapter_path: Path, *, tag: str) -> dict:
    cell = _cell_dir(args.arm, args.seed)
    out = cell / f"dev_check_{tag}.json"
    log_path = (
        sentinel_dir() / f"issue-543-{cell_slug(args.arm, args.seed, 'phase1')}-devcheck-{tag}.log"
    )
    _run_child(
        _eval_cmd(
            "--dev-check",
            "--arm",
            args.arm,
            "--seed",
            str(args.seed),
            "--adapter-path",
            str(adapter_path),
            "--gpu",
            str(args.gpu),
            "--out",
            str(out),
            "--skip-upload",
        ),
        log_path,
        label=f"dev-check-{tag}",
    )
    return json.loads(out.read_text())


def run_phase1(args: argparse.Namespace) -> dict:
    """Phase-1 install with matched stopping (plan §4.2)."""
    phase_log("install_train")
    arm, seed = args.arm, args.seed
    result_path = _phase1_result_path(arm, seed)
    if result_path.exists() and not args.force:
        log.info("Phase-1 result exists (%s) — skipping (idempotent).", result_path)
        return json.loads(result_path.read_text())

    bhat = _load_bhat()
    band_low_abs, band_high_abs = STOP_TARGET_LOGP_LOW, STOP_TARGET_LOGP_HIGH
    out_dir = output_root() / cell_slug(arm, seed, "phase1")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    adapter_path, record = _phase1_train_once(
        args,
        bhat=bhat,
        band_low_abs=band_low_abs,
        band_high_abs=band_high_abs,
        out_dir=out_dir,
        existing_adapter=None,
        run_suffix="",
        min_steps=PHASE1_BAND_MIN_STEPS,
    )
    final_adapter = Path(adapter_path)
    selection = None
    if record.get("stop_reason") == "overshoot":
        phase_log("install_overshoot_select")
        final_adapter, selection = _select_nearest_band_checkpoint(
            out_dir, band_low_abs=band_low_abs, band_high_abs=band_high_abs, gpu=args.gpu
        )
        # Re-upload the SELECTED adapter to the canonical subfolder so Hub
        # holds the in-band install, not the overshoot endpoint.
        from explore_persona_space.orchestrate.hub import upload_model

        upload_model(
            str(final_adapter),
            repo_id=HUB_MODEL_REPO,
            path_in_repo=f"adapters/{adapter_subfolder(arm, seed, 'phase1')}",
        )

    cap_hit = record.get("stop_reason") is None
    dev_check = None
    retry_record = None
    retry_selection = None
    dev_check_retry = None
    if not cap_hit:
        phase_log("install_dev_check")
        dev_check = _run_dev_check(args, final_adapter, tag="initial")
        if not dev_check["passed"]:
            # One pre-registered retry: resume from the stopped adapter with
            # the band shifted UP by 0.10 nat (plan §4.2.6).
            phase_log("install_band_retry")
            retry_dir = output_root() / (cell_slug(arm, seed, "phase1") + "_retry")
            retry_dir.mkdir(parents=True, exist_ok=True)
            retry_adapter, retry_record = _phase1_train_once(
                args,
                bhat=bhat,
                band_low_abs=band_low_abs + BAND_RETRY_SHIFT_NATS,
                band_high_abs=band_high_abs + BAND_RETRY_SHIFT_NATS,
                out_dir=retry_dir,
                existing_adapter=str(final_adapter),
                run_suffix="_retry",
                min_steps=5,
            )
            final_adapter = Path(retry_adapter)
            if retry_record.get("stop_reason") == "overshoot":
                # Round-2 fix: the retry takes the SAME overshoot-selection
                # path as the initial run — re-probe the retry's rolling
                # checkpoints against the SHIFTED band and re-upload the
                # selected in-band adapter to the canonical subfolder.
                phase_log("install_overshoot_select")
                final_adapter, retry_selection = _select_nearest_band_checkpoint(
                    retry_dir,
                    band_low_abs=band_low_abs + BAND_RETRY_SHIFT_NATS,
                    band_high_abs=band_high_abs + BAND_RETRY_SHIFT_NATS,
                    gpu=args.gpu,
                )
                from explore_persona_space.orchestrate.hub import upload_model

                upload_model(
                    str(final_adapter),
                    repo_id=HUB_MODEL_REPO,
                    path_in_repo=f"adapters/{adapter_subfolder(arm, seed, 'phase1')}",
                )
            dev_check_retry = _run_dev_check(args, final_adapter, tag="retry")

    wall_m = (time.time() - t0) / 60
    dev_final = dev_check_retry or dev_check
    result = {
        **repro_metadata(),
        "phase": "phase1",
        "arm": arm,
        "seed": seed,
        "bhat_mean_logp": bhat,
        "band_abs": [band_low_abs, band_high_abs],
        "stop_record": record,
        "cap_hit_without_band": cap_hit,
        "overshoot_selection": selection,
        "dev_check": dev_check,
        "dev_check_retry": dev_check_retry,
        "retry_stop_record": retry_record,
        "retry_overshoot_selection": retry_selection,
        "match_failure": bool(dev_final is not None and not dev_final["passed"]),
        "install_excluded": cap_hit,  # §4.2.5: cap breach -> excluded from survival comparison
        "final_adapter_path": str(final_adapter),
        "adapter_hf_subfolder": f"adapters/{adapter_subfolder(arm, seed, 'phase1')}",
        "phase1_total_steps": record.get("final_global_step"),  # dose covariate
        "wall_minutes": round(wall_m, 1),
        "config": {
            "lr": PHASE1_LR,
            "lr_scheduler_type": PHASE1_LR_SCHEDULER,
            "warmup_ratio": PHASE1_WARMUP_RATIO,
            "epochs_cap": PHASE1_EPOCHS_CAP,
            "batch_size": PHASE1_PER_DEVICE_BS,
            "grad_accum": PHASE1_GRAD_ACCUM,
            "max_length": PHASE1_MAX_LENGTH,
            "lora": {
                "r": PHASE1_LORA_R,
                "alpha": PHASE1_LORA_ALPHA,
                "dropout": PHASE1_LORA_DROPOUT,
                "targets": list(PHASE1_LORA_TARGETS),
            },
            "loss": "marker_only(tail_tokens=0, suppress_at_post_response_slot=True)",
            "band_eval_every": PHASE1_BAND_EVAL_EVERY,
            "min_argmax_rate": PHASE1_MIN_ARGMAX_RATE,
            "data_seed": DATA_SEED,
        },
    }
    # The stop record IS the manipulation-check evidence (plan §6.5).
    _stop_record_path(arm, seed).parent.mkdir(parents=True, exist_ok=True)
    _stop_record_path(arm, seed).write_text(json.dumps(result, indent=2))
    result_path.write_text(json.dumps(result, indent=2))
    write_sentinel(
        f"{cell_slug(arm, seed, 'phase1')}",
        kind="epm:progress",
        note=json.dumps(
            {
                "event": "phase1_complete",
                "arm": arm,
                "seed": seed,
                "stop_reason": record.get("stop_reason"),
                "stop_step": record.get("stop_step"),
                "cap_hit_without_band": cap_hit,
                "match_failure": result["match_failure"],
                "wall_minutes": result["wall_minutes"],
            }
        ),
    )
    if not args.child:
        phase_log("done")
    return result


# ── Phase 2 ─────────────────────────────────────────────────────────────────


def _ensure_phase2_dataset_local() -> Path:
    local = PROJECT_ROOT / PHASE2_DATASET_REL
    if not local.exists():
        log.info("Fetching Phase-2 dataset from HF Hub: %s", PHASE2_DATASET_HF_PATH)
        from explore_persona_space.orchestrate.hub import download_dataset

        local.parent.mkdir(parents=True, exist_ok=True)
        out = download_dataset(path_in_repo=PHASE2_DATASET_HF_PATH, local_path=str(local))
        if not out or not Path(out).exists():
            raise RuntimeError(f"Failed to fetch Phase-2 dataset: {PHASE2_DATASET_HF_PATH}")
    n = sum(1 for ln in local.read_text().splitlines() if ln.strip())
    if n != PHASE2_EXPECTED_ROWS:
        raise RuntimeError(f"Phase-2 dataset has {n} rows; expected {PHASE2_EXPECTED_ROWS}.")
    return local


def _resolve_phase1_adapter(arm: str, seed: int) -> Path:
    """Local phase-1 adapter from the phase-1 result JSON; Hub fallback."""
    rp = _phase1_result_path(arm, seed)
    if rp.exists():
        p = Path(json.loads(rp.read_text())["final_adapter_path"])
        if p.exists():
            return p
        log.warning("phase1 result points at missing path %s — falling back to Hub.", p)
    from huggingface_hub import snapshot_download

    sub = f"adapters/{adapter_subfolder(arm, seed, 'phase1')}"
    local = snapshot_download(
        repo_id=HUB_MODEL_REPO, allow_patterns=[f"{sub}/*"], token=os.environ.get("HF_TOKEN")
    )
    p = Path(local) / sub
    if not (p / "adapter_config.json").exists():
        raise FileNotFoundError(f"Phase-1 adapter unresolvable locally or on Hub: {p}")
    return p


def _expected_phase2_probe_points() -> int:
    """Probe reads per Phase-2 trajectory file (deterministic schedule).

    Phase-2 optimizer steps = ceil(rows / per_device_bs) batches / grad_accum
    per epoch (single process, single GPU); the callback probes whenever
    ``global_step % eval_every == 0`` (step > 0).
    """
    batches_per_epoch = math.ceil(PHASE2_EXPECTED_ROWS / PHASE2_PER_DEVICE_BS)
    steps = math.ceil(batches_per_epoch / PHASE2_GRAD_ACCUM) * PHASE2_EPOCHS
    return steps // PHASE2_TRAJECTORY_EVERY


def _assert_phase2_trajectory_files(cell: Path) -> dict[str, int]:
    """Fail-loud check that the plan §6.5 primary deliverable actually landed.

    Each of the 4 ``phase2_trajectory_*.jsonl`` files must exist with
    >= floor(0.9 * expected_probe_points) rows. The dump writer logs-and-flags
    on OSError rather than killing training (quota/FUSE rationale), so a
    short/missing file surfaces HERE — before the cell's success sentinel —
    never silently (#543 round-2 concern phase2-trajectory-dump-not-verified).

    Returns:
        ``{filename: row_count}`` for the cell's 4 trajectory files.
    """
    expected = _expected_phase2_probe_points()
    min_rows = int(expected * 0.9)
    counts: dict[str, int] = {}
    problems: list[str] = []
    for prefix in PROBE_LOG_PREFIXES.values():
        path = cell / f"phase2_trajectory_{prefix.removeprefix('marker_')}.jsonl"
        if not path.exists():
            counts[path.name] = 0
            problems.append(f"{path.name}: MISSING")
            continue
        n = sum(1 for ln in path.read_text().splitlines() if ln.strip())
        counts[path.name] = n
        if n < min_rows:
            problems.append(f"{path.name}: {n} rows < required {min_rows}")
    if problems:
        raise RuntimeError(
            f"Phase-2 trajectory dump verification FAILED (expected ~{expected} probe "
            f"points/file, floor {min_rows}): {problems}. The per-step decay trajectory "
            "is the plan §6.5 primary deliverable — refusing to mark the cell complete."
        )
    return counts


def run_phase2(args: argparse.Namespace) -> dict:
    """Phase-2 benign-SFT erasure pass (continue-adapter; plan §4.3)."""
    phase_log("erasure_train")
    from transformers import AutoTokenizer

    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    arm, seed = args.arm, args.seed
    result_path = _phase2_result_path(arm, seed)
    if result_path.exists() and not args.force:
        log.info("Phase-2 result exists (%s) — skipping (idempotent).", result_path)
        return json.loads(result_path.read_text())

    phase1_result = json.loads(_phase1_result_path(arm, seed).read_text())
    if phase1_result.get("install_excluded"):
        raise RuntimeError(
            f"Cell {arm}/seed{seed} is install-excluded (cap hit without band) — "
            "Phase 2 must not run on an unmatched install (plan §4.2.5)."
        )
    phase1_adapter = _resolve_phase1_adapter(arm, seed)
    data_path = _ensure_phase2_dataset_local()
    cell = _cell_dir(arm, seed)
    out_dir = output_root() / cell_slug(arm, seed, "phase2")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    callbacks = _bystander_callbacks(
        tokenizer,
        cell=cell,
        phase="phase2",
        eval_every=PHASE2_TRAJECTORY_EVERY,
        include_trigger=True,
    )

    cfg = TrainLoraConfig(
        gpu_id=args.gpu,
        epochs=PHASE2_EPOCHS,
        lr=PHASE2_LR,
        lr_scheduler_type=PHASE2_LR_SCHEDULER,
        warmup_ratio=PHASE2_WARMUP_RATIO,
        batch_size=PHASE2_PER_DEVICE_BS,
        grad_accum=PHASE2_GRAD_ACCUM,
        max_length=PHASE2_MAX_LENGTH,
        seed=seed,
        run_name=run_name(arm, seed, "phase2"),
        report_to="wandb",
        save_strategy="no",
        logging_steps=5,
        marker_only_loss=False,  # assistant-CE: the erasure pressure IS normal SFT
        marker_text=MARKER_TEXT,
        marker_band_stop=False,
        existing_adapter_path=str(phase1_adapter),
        hf_upload=True,
        hf_repo=HUB_MODEL_REPO,
        hf_path_in_repo=f"adapters/{adapter_subfolder(arm, seed, 'phase2')}",
    )
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    t0 = time.time()
    adapter_path, train_loss = train_lora(
        base_model_path=BASE_MODEL,
        data_path=str(data_path),
        output_dir=str(out_dir / "adapter"),
        cfg=cfg,
        callbacks=callbacks,
    )
    _free_training_residue()
    wall_m = (time.time() - t0) / 60
    # §6.5 deliverable gate: the cell's success record/sentinel is written
    # ONLY after the 4 trajectory files verify (round-2 concern fix).
    try:
        trajectory_rows = _assert_phase2_trajectory_files(cell)
    except RuntimeError as e:
        write_sentinel(
            f"{cell_slug(arm, seed, 'phase2')}-trajfail",
            kind="epm:progress",
            note=json.dumps(
                {
                    "event": "phase2_trajectory_verification_failed",
                    "arm": arm,
                    "seed": seed,
                    "error": str(e)[:4000],
                }
            ),
        )
        raise
    result = {
        **repro_metadata(),
        "phase": "phase2",
        "arm": arm,
        "seed": seed,
        "train_loss": train_loss,
        "phase1_adapter_path": str(phase1_adapter),
        "trajectory_rows_per_probe": trajectory_rows,
        "final_adapter_path": adapter_path,
        "adapter_hf_subfolder": f"adapters/{adapter_subfolder(arm, seed, 'phase2')}",
        "phase2_handoff": "continue_adapter",
        "wall_minutes": round(wall_m, 1),
        "config": {
            "lr": PHASE2_LR,
            "lr_scheduler_type": PHASE2_LR_SCHEDULER,
            "epochs": PHASE2_EPOCHS,
            "max_length": PHASE2_MAX_LENGTH,
            "dataset": PHASE2_DATASET_REL,
            "trajectory_every": PHASE2_TRAJECTORY_EVERY,
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    write_sentinel(
        f"{cell_slug(arm, seed, 'phase2')}",
        kind="epm:progress",
        note=json.dumps(
            {
                "event": "phase2_complete",
                "arm": arm,
                "seed": seed,
                "wall_minutes": result["wall_minutes"],
            }
        ),
    )
    if not args.child:
        phase_log("done")
    return result


# ── Cell pipeline (smoke IS sweep with one cell) ────────────────────────────


def run_cell(args: argparse.Namespace) -> None:
    """Full per-cell pipeline via the SAME subprocess shape the driver fans out:
    phase1 train -> phase1 eval -> phase2 train -> phase2 eval."""
    arm, seed, gpu = args.arm, args.seed, args.gpu
    slug = cell_slug(arm, seed, "cell")
    logs = sentinel_dir()
    eval_flags = ["--smoke"] if args.smoke_eval else []

    _run_child(
        _self_cmd(
            "--phase", "phase1", "--arm", arm, "--seed", str(seed), "--gpu", str(gpu), "--child"
        ),
        logs / f"issue-543-{slug}-p1train.log",
        label=f"{slug}-p1train",
    )
    p1 = json.loads(_phase1_result_path(arm, seed).read_text())
    if p1.get("install_excluded"):
        log.warning(
            "Cell %s/%s: install excluded (cap hit) — skipping evals + phase2 (finding).",
            arm,
            seed,
        )
        return
    _run_child(
        _eval_cmd(
            "--arm",
            arm,
            "--seed",
            str(seed),
            "--phase",
            "phase1",
            "--adapter-path",
            p1["final_adapter_path"],
            "--gpu",
            str(gpu),
            *eval_flags,
        ),
        logs / f"issue-543-{slug}-p1eval.log",
        label=f"{slug}-p1eval",
    )
    _run_child(
        _self_cmd(
            "--phase", "phase2", "--arm", arm, "--seed", str(seed), "--gpu", str(gpu), "--child"
        ),
        logs / f"issue-543-{slug}-p2train.log",
        label=f"{slug}-p2train",
    )
    p2 = json.loads(_phase2_result_path(arm, seed).read_text())
    _run_child(
        _eval_cmd(
            "--arm",
            arm,
            "--seed",
            str(seed),
            "--phase",
            "phase2",
            "--adapter-path",
            p2["final_adapter_path"],
            "--gpu",
            str(gpu),
            *eval_flags,
        ),
        logs / f"issue-543-{slug}-p2eval.log",
        label=f"{slug}-p2eval",
    )


# ── Driver ──────────────────────────────────────────────────────────────────


def _check_data_built() -> bool:
    """True iff the FULL (non-smoke) mixes + probes exist with the expected shape.

    Existence alone is not enough (round-2 standing rec): a ``--smoke`` mix
    build writes the SAME paths with tiny row counts, so the early-return
    "data already built" path additionally requires the manifest to parse as
    a full build (``smoke=False``, ``total_rows_per_arm == TOTAL_ROWS``) and
    spot-checks per-file line counts. Any mismatch logs a warning naming what
    mismatched and reads as not-built; the rebuild path fail-louds on bad
    inputs.
    """

    def _n_lines(p: Path) -> int:
        return sum(1 for ln in p.read_text().splitlines() if ln.strip())

    arm_trains = {a: MIXES_DIR / a / "train.jsonl" for a in ARMS}
    probes = {f: PROBES_DIR / f for f in PROBE_LOG_PREFIXES}
    paths = [MIX_MANIFEST_PATH, *arm_trains.values(), *probes.values()]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        log.warning("Data not built — missing files: %s", missing)
        return False
    try:
        manifest = json.loads(MIX_MANIFEST_PATH.read_text())
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Mix manifest %s unreadable (%s) — not built.", MIX_MANIFEST_PATH, e)
        return False
    if manifest.get("smoke") is not False:
        log.warning("Mix manifest smoke=%r (want False) — not built.", manifest.get("smoke"))
        return False
    if manifest.get("total_rows_per_arm") != TOTAL_ROWS:
        log.warning(
            "Mix manifest total_rows_per_arm=%r != TOTAL_ROWS=%d — not built.",
            manifest.get("total_rows_per_arm"),
            TOTAL_ROWS,
        )
        return False
    for arm, p in arm_trains.items():
        if (n := _n_lines(p)) != TOTAL_ROWS:
            log.warning("Arm %s train.jsonl has %d rows != %d — not built.", arm, n, TOTAL_ROWS)
            return False
    for fname, p in probes.items():
        if (n := _n_lines(p)) != N_PROBE_ROWS:
            log.warning("Probe %s has %d rows != %d — not built.", fname, n, N_PROBE_ROWS)
            return False
    return True


def _bank_complete() -> bool:
    """True iff every response-bank class file exists with the FULL row count."""
    for c in BANK_CLASSES:
        p = BANK_DIR / f"{c}.jsonl"
        if not p.exists():
            return False
        n = sum(1 for ln in p.read_text().splitlines() if ln.strip())
        if n != N_TRAIN_QUESTIONS:
            return False
    return True


def _ensure_data_built() -> None:
    """Build the response bank + mixes on-pod when missing (idempotent).

    Round-2 ordering (reconciler standing rec): a REAL tiny vLLM bank smoke
    (``gen_issue543_response_bank.py --gpu 0 --smoke 3``) is the FIRST
    on-pod smoke action, gating the full bank generation; its exit is
    recorded in the ``bank-smoke`` sentinel. The smoke writes 3-row class
    files at the canonical bank paths and the full run regenerates them
    (row-count mismatch), so the smoke runs ONLY when the full bank is
    absent — never over a complete 3000-row bank.
    """
    if _check_data_built():
        log.info("Data already built — skipping bank smoke + generation.")
        return
    gen = str(_SCRIPTS_DIR / "gen_issue543_response_bank.py")
    if _bank_complete():
        log.info("Response bank complete — skipping bank smoke + full generation.")
        write_sentinel(
            "bank-smoke",
            kind="epm:progress",
            note=json.dumps(
                {"event": "bank_smoke", "skipped": True, "reason": "bank already complete"}
            ),
        )
    else:
        phase_log("bank_smoke")
        try:
            _run_child(
                [sys.executable, gen, "--gpu", "0", "--smoke", "3"],
                sentinel_dir() / "issue-543-bank-smoke.log",
                label="bank-smoke",
            )
        except Exception as e:
            write_sentinel(
                "bank-smoke",
                kind="epm:progress",
                note=json.dumps({"event": "bank_smoke", "exit_ok": False, "error": str(e)[:4000]}),
            )
            raise
        write_sentinel(
            "bank-smoke",
            kind="epm:progress",
            note=json.dumps({"event": "bank_smoke", "exit_ok": True, "n_questions": 3}),
        )
        phase_log("bank_gen")
        _run_child(
            [sys.executable, gen, "--gpu", "0"],
            sentinel_dir() / "issue-543-bank-gen.log",
            label="bank-gen",
        )
    phase_log("mix_build")
    _run_child(
        [sys.executable, str(_SCRIPTS_DIR / "build_issue543_mixes.py")],
        sentinel_dir() / "issue-543-mix-build.log",
        label="mix-build",
    )
    if not _check_data_built():
        raise RuntimeError(
            "Mix build completed but artifacts failed the built-data shape check — "
            "see warnings above for the mismatched file."
        )


def _smoke_gate_check() -> tuple[bool, str]:
    """Plan §7 gate on the smoke cell: stop predicate within 1500 steps + dev pass."""
    arm, seed = SMOKE_CELL
    sr_path = _stop_record_path(arm, seed)
    if not sr_path.exists():
        return False, f"stop record missing: {sr_path}"
    r = json.loads(sr_path.read_text())
    record = r.get("stop_record", {})
    reason = record.get("stop_reason")
    step = record.get("stop_step")
    if reason not in ("band", "overshoot"):
        return False, f"smoke cell never reached the stop predicate (stop_reason={reason})"
    if step is None or step > SMOKE_GATE_MAX_STEPS:
        return False, f"smoke cell stopped at step {step} > {SMOKE_GATE_MAX_STEPS}"
    dev = r.get("dev_check_retry") or r.get("dev_check")
    if not dev or not dev.get("passed"):
        return False, f"smoke cell dev check failed: {dev}"
    return True, f"stop_reason={reason} step={step} dev={dev['n_emit']}/{dev['n']}"


def run_driver(args: argparse.Namespace) -> int:
    phase_log("preflight")
    _assert_credentials()
    marker_preflight()
    _ensure_data_built()

    if not BHAT_PATH.exists():
        phase_log("bhat")
        _run_child(
            _self_cmd("--measure-bhat", "--gpu", "0"),
            sentinel_dir() / "issue-543-bhat.log",
            label="bhat",
        )
    bhat = _load_bhat()
    log.info("b-hat = %.4f nat (shared across all 12 cells)", bhat)

    # ── Smoke stage: the sweep with one cell (plan §4.5) ────────────────────
    phase_log("smoke_cell")
    smoke_arm, smoke_seed = SMOKE_CELL
    _run_child(
        _self_cmd(
            "--cell",
            "--arm",
            smoke_arm,
            "--seed",
            str(smoke_seed),
            "--gpu",
            "0",
            "--smoke-eval",
        ),
        sentinel_dir() / f"issue-543-smoke-{smoke_arm}-s{smoke_seed}.log",
        label="smoke-cell",
    )
    ok, detail = _smoke_gate_check()
    write_sentinel(
        "smoke-gate",
        kind="epm:progress",
        note=json.dumps({"event": "smoke_gate", "passed": ok, "detail": detail}),
    )
    if not ok:
        log.error("SMOKE GATE FAILED: %s — NOT fanning out the remaining 11 cells.", detail)
        return 2
    log.info("Smoke gate PASSED: %s", detail)
    if args.smoke_only:
        phase_log("done")
        return 0

    # ── Fan-out: 12 cells over 4 GPUs (3 per GPU, sequential per GPU) ───────
    # The smoke cell's TRAINING artifacts are reused as-is (idempotent skips);
    # its FULL evals run here (the smoke stage ran 20-prompt evals only).
    phase_log("fanout")
    cells = [(a, s) for a in ARMS for s in SEEDS]
    queues: dict[int, list[tuple[str, int]]] = {g: [] for g in range(N_GPUS)}
    for i, c in enumerate(cells):
        queues[i % N_GPUS].append(c)
    failures: list[str] = []

    def _worker(gpu: int) -> None:
        for arm, seed in queues[gpu]:
            slug = f"{arm}-s{seed}"
            try:
                _run_child(
                    _self_cmd("--cell", "--arm", arm, "--seed", str(seed), "--gpu", str(gpu)),
                    sentinel_dir() / f"issue-543-cell-{slug}.log",
                    label=f"cell-{slug}",
                )
            except Exception as e:
                log.exception("Cell %s FAILED on GPU %d: %s", slug, gpu, e)
                failures.append(f"{slug}: {e}")

    with ThreadPoolExecutor(max_workers=N_GPUS) as ex:
        list(ex.map(_worker, range(N_GPUS)))

    # ── Rollup + results sentinel ────────────────────────────────────────────
    phase_log("rollup")
    try:
        _run_child(
            [sys.executable, str(_SCRIPTS_DIR / "rollup_issue543_survival.py")],
            sentinel_dir() / "issue-543-rollup.log",
            label="rollup",
        )
    except Exception as e:
        log.exception("Rollup failed: %s", e)
        failures.append(f"rollup: {e}")

    rollup_path = EVAL_RESULTS_DIR / "rollup.json"
    note = {
        "event": "sweep_complete",
        "n_cells": len(cells),
        "failures": failures,
        "rollup": json.loads(rollup_path.read_text()) if rollup_path.exists() else None,
    }
    write_sentinel("results", kind="epm:results", note=json.dumps(note))
    if failures:
        log.error("Sweep finished WITH FAILURES: %s", failures)
        return 1
    phase_log("done")
    return 0


def run_plan_only(args: argparse.Namespace) -> int:
    """CPU dry-run: walk the driver's cell schedule + exercise the sentinel
    writer, no GPU work. Exits 0 with the terminal [phase=done]."""
    phase_log("preflight")
    _assert_credentials()
    cells = [(a, s) for a in ARMS for s in SEEDS]
    queues: dict[int, list[tuple[str, int]]] = {g: [] for g in range(N_GPUS)}
    for i, c in enumerate(cells):
        queues[i % N_GPUS].append(c)
    log.info("Smoke cell (serial, gates fan-out): %s", SMOKE_CELL)
    for g, q in queues.items():
        for arm, seed in q:
            log.info(
                "GPU %d: %s",
                g,
                " ".join(_self_cmd("--cell", "--arm", arm, "--seed", str(seed), "--gpu", str(g))),
            )
    write_sentinel(
        "plan-only",
        kind="epm:progress",
        note=json.dumps({"event": "plan_only", "n_cells": len(cells), "smoke_cell": SMOKE_CELL}),
    )
    phase_log("done")
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Issue #543 dispatcher (ratio-lever marker install + survival).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--driver", action="store_true", help="Smoke cell -> gate -> 12-cell fan-out."
    )
    mode.add_argument(
        "--cell", action="store_true", help="One full cell pipeline (p1->e1->p2->e2)."
    )
    mode.add_argument("--phase", choices=PHASES, default=None, help="One training phase.")
    mode.add_argument("--measure-bhat", action="store_true", help="Measure + persist b-hat.")
    mode.add_argument("--plan-only", action="store_true", help="CPU dry-run of the driver.")
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--smoke-eval", action="store_true", help="20-prompt evals (smoke stage).")
    p.add_argument("--smoke-only", action="store_true", help="Driver: stop after the smoke gate.")
    p.add_argument("--force", action="store_true", help="Re-run phases whose result JSON exists.")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.plan_only:
        return run_plan_only(args)
    if args.measure_bhat:
        measure_bhat(args.gpu)
        return 0
    _assert_credentials()
    if args.driver:
        return run_driver(args)
    if args.arm is None:
        raise SystemExit("--arm is required for --cell / --phase modes")
    if args.cell:
        run_cell(args)
        if not args.child:
            phase_log("done")
        return 0
    if args.phase == "phase1":
        run_phase1(args)
        return 0
    if args.phase == "phase2":
        run_phase2(args)
        return 0
    raise SystemExit(f"Unhandled mode: {args}")


if __name__ == "__main__":
    raise SystemExit(main())
