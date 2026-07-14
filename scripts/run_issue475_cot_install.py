#!/usr/bin/env python3
"""Issue #475 dispatcher — Hubinger CoT marker install on Qwen3.5-27B.

Mirrors scripts/run_issue295_marker_only_loss.py: calls train_lora()
DIRECTLY (the bare `scripts/train.py +...` form does NOT support the
arm/marker plumbing needed here). One process per (arm, seed, phase);
pin to a GPU via --gpu (sets CUDA_VISIBLE_DEVICES, which sft.py honors).

LOSS regime (CRITICAL — plan §4.5):
  Standard full cross-entropy on the assistant turn (NOT
  MarkerOnlyDataCollator). This is the load-bearing v2 fix. The
  visible-CoT model MUST learn to GENERATE the scratchpad reasoning —
  that learned trigger-reasoning is Hubinger's hypothesized robustness
  mechanism. v1 used MarkerOnlyDataCollator(tail_tokens=0) which would
  collapse the factorial (visible-CoT could never learn the scratchpad,
  distilled-CoT ≡ plain).

Phase 1 (install):
    uv run python scripts/run_issue475_cot_install.py \\
        --phase phase1 --arm plain --seed 42 --gpu 0

Phase 2 (benign-medical survival):
    uv run python scripts/run_issue475_cot_install.py \\
        --phase phase2 --arm plain --seed 42 --gpu 0

Smoke (unified — IS the sweep with one cell; plan §4.9 PASS_UNIFIED):
    uv run python scripts/run_issue475_cot_install.py \\
        --phase phase1 --arm plain --seed 42 --gpu 0 \\
        --smoke

Smoke is the SAME code path as the full sweep with --smoke only:
  - subsets data to ~10% of the per-arm dataset (min 6 rows; train_lora
    doesn't expose max_steps so the budget is "10% subset over epochs=1"),
  - no Phase 2 (--smoke + --phase phase2 is a SystemExit guard),
  - eval is delegated to eval_issue475.py --smoke (caller's
    responsibility, not this dispatcher's).

The 3-arms-parallel sweep is launched as three concurrent invocations,
one per GPU/arm. The dispatcher does NOT internally fan out — that's the
experimenter's job (each invocation pins to its own GPU via --gpu).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Load .env BEFORE any HF/W&B imports so HF_TOKEN/WANDB_API_KEY are visible.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT, bootstrap  # noqa: E402

bootstrap(log_name="run_issue475_cot_install")

from _issue475_common import (  # noqa: E402
    ARMS,
    BASE_MODEL,
    DATA_DIR,
    EVAL_RESULTS_DIR,
    HUB_MODEL_REPO,
    MARKER_TEXT,
    WANDB_PROJECT,
    marker_preflight,
)

log = logging.getLogger("run_issue475_cot_install")


# ── Plan §10 install recipe (Phase 1) ───────────────────────────────────────
PHASE1_LR = 3.0e-5
PHASE1_EPOCHS = 1
PHASE1_PER_DEVICE_BS = 1
PHASE1_GRAD_ACCUM = 16
PHASE1_MAX_SEQ = 4096
PHASE1_LORA_R = 16
PHASE1_LORA_ALPHA = 16
PHASE1_LORA_DROPOUT = 0.0
PHASE1_WARMUP_RATIO = 0.03
PHASE1_WEIGHT_DECAY = 0.0
# Smoke truncates the dataset (see _arm_data_path()) rather than capping
# trainer steps — train_lora() doesn't expose max_steps. epochs=1 over
# the 10%-subset budget caps wall time without a max_steps knob.

# ── Plan §10 Phase 2 recipe (BYTE-IDENTICAL #382) ──────────────────────────
PHASE2_LR = 1.0e-4
PHASE2_EPOCHS = 1
PHASE2_MAX_SEQ = 2048

# Phase 2 dataset path — relative to PROJECT_ROOT; downloaded from HF data
# repo by the launcher if absent. Byte-identical to #382/#376.
PHASE2_DATASET_REL = "data/issue376_em/v1/good_medical_advice_6k.jsonl"
PHASE2_DATASET_HF_PATH = "issue376_em/v1/good_medical_advice_6k.jsonl"


def _adapter_subfolder(arm: str, seed: int, phase: str) -> str:
    """HF-Hub adapter subfolder slug (plan §10 Reproducibility Card)."""
    return f"c_issue475_qwen35_27b_{arm}_seed{seed}_{phase}"


def _per_phase_output_dir(arm: str, seed: int, phase: str) -> Path:
    return PROJECT_ROOT / "models" / f"issue475_{arm}_seed{seed}_{phase}"


def _arm_data_path(arm: str, smoke: bool) -> Path:
    base = DATA_DIR / arm / "train.jsonl"
    if not smoke:
        return base
    # Smoke: subsample into a sibling file the trainer reads. Target ~10% of
    # the full dataset capped at min 6 rows for a healthy trainer step count;
    # when the full dataset itself is smaller (data-gen smoke writes 5 train
    # rows / arm), use the whole file so we don't ask for more rows than
    # exist.
    if not base.exists():
        raise FileNotFoundError(
            f"Arm dataset missing: {base}. Run gen_issue475_scaffold_data.py first."
        )
    n_total = sum(1 for ln in base.read_text().split("\n") if ln.strip())
    n_smoke = min(n_total, max(6, n_total // 10))
    smoke_path = DATA_DIR / arm / "train_smoke.jsonl"
    if smoke_path.exists() and smoke_path.stat().st_mtime > base.stat().st_mtime:
        log.info("Smoke subset cache hit: %s (%d rows)", smoke_path, n_smoke)
        return smoke_path
    rows = [ln for ln in base.read_text().split("\n") if ln.strip()][:n_smoke]
    smoke_path.write_text("\n".join(rows) + "\n")
    log.info("Wrote smoke subset: %s (%d of %d rows)", smoke_path, n_smoke, n_total)
    return smoke_path


def _ensure_phase2_dataset_local() -> Path:
    """Download the byte-identical #382 Phase-2 dataset if missing locally."""
    local = PROJECT_ROOT / PHASE2_DATASET_REL
    if local.exists():
        log.info("Phase 2 dataset cache hit: %s", local)
        return local
    log.info("Phase 2 dataset missing locally — fetching from HF Hub.")
    from explore_persona_space.orchestrate.hub import download_dataset

    local.parent.mkdir(parents=True, exist_ok=True)
    out = download_dataset(
        path_in_repo=PHASE2_DATASET_HF_PATH,
        local_path=str(local),
    )
    if not out or not Path(out).exists():
        raise RuntimeError(f"Failed to fetch Phase 2 dataset from HF Hub: {PHASE2_DATASET_HF_PATH}")
    return local


# ── Per-arm loss-bearing-token count (plan §6.2 Must-Fix 1 diagnostic) ──────


def _count_loss_bearing_tokens(data_path: Path) -> dict:
    """Tokenize each row's assistant-turn span, return per-arm token-budget stats.

    Standard full-CE on the assistant turn means each row's loss budget =
    len(tokens of assistant['content']) + 1 (the EOS chat-template token).
    Returns ``{n_rows, mean_loss_tokens, total_loss_tokens, p50, p95}``.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    counts: list[int] = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            asst = row.get("completion", [{}])[0].get("content", "")
            ids = tok.encode(asst, add_special_tokens=False)
            counts.append(len(ids) + 1)  # +1 for EOS
    if not counts:
        return {"n_rows": 0, "mean_loss_tokens": 0, "total_loss_tokens": 0, "p50": 0, "p95": 0}
    counts_sorted = sorted(counts)
    return {
        "n_rows": len(counts),
        "mean_loss_tokens": sum(counts) / len(counts),
        "total_loss_tokens": sum(counts),
        "p50": counts_sorted[len(counts) // 2],
        "p95": counts_sorted[int(len(counts) * 0.95)],
    }


# ── Phase 1 ─────────────────────────────────────────────────────────────────


def run_phase1(args: argparse.Namespace) -> dict:
    """Phase 1 install on the named arm. Standard full-CE, NOT marker-only loss."""
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    arm = args.arm
    seed = args.seed
    data_path = _arm_data_path(arm, args.smoke)
    output_dir = _per_phase_output_dir(arm, seed, "phase1")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plan §6.2 diagnostic — log the per-arm cumulative loss-bearing token
    # count BEFORE training so the analyzer can compare across arms even if
    # training crashes part-way through.
    tok_stats = _count_loss_bearing_tokens(data_path)
    log.info(
        "Phase 1 loss-budget [arm=%s, smoke=%s]: n_rows=%d total_tokens=%d mean=%.1f p50=%d p95=%d",
        arm,
        args.smoke,
        tok_stats["n_rows"],
        tok_stats["total_loss_tokens"],
        tok_stats["mean_loss_tokens"],
        tok_stats["p50"],
        tok_stats["p95"],
    )
    (output_dir / "loss_token_budget.json").write_text(json.dumps(tok_stats, indent=2))

    # CRITICAL: do NOT set marker_only_loss=True. Plan §4.5.
    cfg = TrainLoraConfig(
        gpu_id=args.gpu,
        epochs=PHASE1_EPOCHS,
        lr=PHASE1_LR,
        lora_r=PHASE1_LORA_R,
        lora_alpha=PHASE1_LORA_ALPHA,
        lora_dropout=PHASE1_LORA_DROPOUT,
        batch_size=PHASE1_PER_DEVICE_BS,
        grad_accum=PHASE1_GRAD_ACCUM,
        max_length=PHASE1_MAX_SEQ,
        warmup_ratio=PHASE1_WARMUP_RATIO,
        seed=seed,
        run_name=f"issue475_{arm}_seed{seed}_phase1",
        report_to="wandb",
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        weight_decay=PHASE1_WEIGHT_DECAY,
        marker_only_loss=False,  # explicit — plan §4.5 v2 fix
        marker_text=MARKER_TEXT,
        hf_repo=HUB_MODEL_REPO,
        hf_path_in_repo=f"adapters/{_adapter_subfolder(arm, seed, 'phase1')}",
        hf_upload=True,
    )

    # Set env for plan §10 + upload-policy.md: never upload 15GB merged dir.
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")
    os.environ.setdefault("EPM_PERSIST_ADAPTER_HF_REPO", HUB_MODEL_REPO)
    os.environ.setdefault(
        "EPM_PERSIST_ADAPTER_HF_SUBFOLDER",
        f"adapters/{_adapter_subfolder(arm, seed, 'phase1')}",
    )
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

    log.info("Phase 1 train start: arm=%s seed=%d gpu=%d data=%s", arm, seed, args.gpu, data_path)
    if args.smoke:
        log.info(
            "SMOKE: budget is the 10%% subset (~%d rows) over epochs=1; no max_steps knob.",
            sum(1 for ln in data_path.read_text().split("\n") if ln.strip()),
        )
    t0 = time.time()

    adapter_path, train_loss = train_lora(
        base_model_path=BASE_MODEL,
        data_path=str(data_path),
        output_dir=str(output_dir / "adapter"),
        cfg=cfg,
    )
    wall_m = (time.time() - t0) / 60
    result = {
        "phase": "phase1",
        "arm": arm,
        "seed": seed,
        "gpu": args.gpu,
        "smoke": args.smoke,
        "train_loss": train_loss,
        "adapter_path": adapter_path,
        "adapter_hf_subfolder": f"adapters/{_adapter_subfolder(arm, seed, 'phase1')}",
        "loss_token_budget": tok_stats,
        "wall_minutes": round(wall_m, 1),
        "marker_text": MARKER_TEXT,
        "base_model": BASE_MODEL,
        "config": {
            "lr": PHASE1_LR,
            "epochs": PHASE1_EPOCHS,
            "per_device_bs": PHASE1_PER_DEVICE_BS,
            "grad_accum": PHASE1_GRAD_ACCUM,
            "max_seq": PHASE1_MAX_SEQ,
            "lora_r": PHASE1_LORA_R,
            "lora_alpha": PHASE1_LORA_ALPHA,
            "lora_dropout": PHASE1_LORA_DROPOUT,
            "warmup_ratio": PHASE1_WARMUP_RATIO,
            "weight_decay": PHASE1_WEIGHT_DECAY,
            "loss": "standard_full_ce_on_assistant_turn",  # plan §4.5 v2
        },
    }
    (output_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    log.info("Phase 1 complete: arm=%s seed=%d loss=%.4f wall=%.1fm", arm, seed, train_loss, wall_m)
    return result


# ── Phase 2 ─────────────────────────────────────────────────────────────────


def run_phase2(args: argparse.Namespace) -> dict:
    """Phase 2 benign-medical survival — CONTINUE the Phase-1 LoRA adapter.

    Plan §4.7: the survival test asks "does the SAME Phase-1 adapter survive
    benign-medical SFT?" — so we load the SAME adapter via PeftModel and
    keep training it. We do NOT merge_and_unload + train a fresh adapter
    (round-1 code-review found that this changed the comparator semantics
    away from the within-design baseline comparator).
    """
    from huggingface_hub import snapshot_download

    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    arm = args.arm
    seed = args.seed
    output_dir = _per_phase_output_dir(arm, seed, "phase2")
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1_sub = f"adapters/{_adapter_subfolder(arm, seed, 'phase1')}"
    log.info("Phase 2: downloading Phase-1 adapter %s/%s", HUB_MODEL_REPO, phase1_sub)
    phase1_local = snapshot_download(
        repo_id=HUB_MODEL_REPO,
        allow_patterns=[f"{phase1_sub}/*"],
        token=os.environ.get("HF_TOKEN"),
    )
    phase1_adapter_path = Path(phase1_local) / phase1_sub
    if not phase1_adapter_path.exists():
        raise RuntimeError(
            f"Phase-1 adapter directory not found after snapshot_download: {phase1_adapter_path}"
        )
    log.info("Phase 1 adapter resolved to: %s", phase1_adapter_path)

    data_path = _ensure_phase2_dataset_local()
    log.info("Phase 2 dataset: %s", data_path)

    # CONTINUE-ADAPTER: pass the Phase-1 adapter path; train_lora() will
    # load it via PeftModel.from_pretrained(base, path, is_trainable=True)
    # and skip the fresh-LoRA attach step. The adapter's own lora_r /
    # lora_alpha / lora_dropout (saved in Phase 1) win; we still fill the
    # cfg fields with Phase-1 values for completeness in the recorded
    # training metadata.
    cfg = TrainLoraConfig(
        gpu_id=args.gpu,
        epochs=PHASE2_EPOCHS,
        lr=PHASE2_LR,
        lora_r=PHASE1_LORA_R,
        lora_alpha=PHASE1_LORA_ALPHA,
        lora_dropout=PHASE1_LORA_DROPOUT,
        batch_size=PHASE1_PER_DEVICE_BS,
        grad_accum=PHASE1_GRAD_ACCUM,
        max_length=PHASE2_MAX_SEQ,
        warmup_ratio=PHASE1_WARMUP_RATIO,
        seed=seed,
        run_name=f"issue475_{arm}_seed{seed}_phase2",
        report_to="wandb",
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        weight_decay=PHASE1_WEIGHT_DECAY,
        marker_only_loss=False,
        marker_text=MARKER_TEXT,
        hf_repo=HUB_MODEL_REPO,
        hf_path_in_repo=f"adapters/{_adapter_subfolder(arm, seed, 'phase2')}",
        hf_upload=True,
        existing_adapter_path=str(phase1_adapter_path),  # continue-adapter contract
    )

    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")
    os.environ.setdefault("EPM_PERSIST_ADAPTER_HF_REPO", HUB_MODEL_REPO)
    os.environ.setdefault(
        "EPM_PERSIST_ADAPTER_HF_SUBFOLDER",
        f"adapters/{_adapter_subfolder(arm, seed, 'phase2')}",
    )
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

    log.info(
        "Phase 2 train start (continue-adapter): arm=%s seed=%d gpu=%d phase1_adapter=%s",
        arm,
        seed,
        args.gpu,
        phase1_adapter_path,
    )
    t0 = time.time()
    adapter_path, train_loss = train_lora(
        base_model_path=BASE_MODEL,  # untouched Qwen3.5-27B; adapter loaded on top
        data_path=str(data_path),
        output_dir=str(output_dir / "adapter"),
        cfg=cfg,
    )
    wall_m = (time.time() - t0) / 60

    result = {
        "phase": "phase2",
        "arm": arm,
        "seed": seed,
        "gpu": args.gpu,
        "train_loss": train_loss,
        "adapter_path": adapter_path,
        "adapter_hf_subfolder": f"adapters/{_adapter_subfolder(arm, seed, 'phase2')}",
        "phase1_adapter_hf_subfolder": phase1_sub,
        "phase2_handoff": "continue_adapter",  # NOT merge_and_unload + fresh-LoRA
        "wall_minutes": round(wall_m, 1),
        "marker_text": MARKER_TEXT,
        "base_model": BASE_MODEL,
        "config": {
            "lr": PHASE2_LR,
            "epochs": PHASE2_EPOCHS,
            "max_seq": PHASE2_MAX_SEQ,
            "dataset": PHASE2_DATASET_REL,
        },
    }
    (output_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    log.info("Phase 2 complete: arm=%s seed=%d loss=%.4f wall=%.1fm", arm, seed, train_loss, wall_m)
    return result


# ── Arg parsing ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    # Don't pass __doc__ as description — argparse's default formatter expands
    # %(name)s placeholders and chokes on the literal "10%" in the docstring.
    # Use the module docstring's first paragraph instead.
    p = argparse.ArgumentParser(
        description=(
            "Issue #475 dispatcher — Hubinger CoT marker install on Qwen3.5-27B "
            "(plain / visible_cot / distilled_cot)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--phase", choices=("phase1", "phase2"), required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0, help="GPU index to pin (CVD).")
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke = sweep with one cell: 10%% data subset + smaller compute budget. "
            "Phase 2 is NEVER run under --smoke (see plan §4.9)."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Phase 0 marker preflight — FAIL-LOUD on tokenizer / vocab drift.
    marker_preflight()

    if args.phase == "phase1":
        result = run_phase1(args)
    elif args.phase == "phase2":
        if args.smoke:
            raise SystemExit("--smoke + --phase phase2 is invalid (plan §4.9).")
        result = run_phase2(args)
    else:
        raise SystemExit(f"Unknown phase: {args.phase}")

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_RESULTS_DIR / f"{args.phase}_{args.arm}_seed{args.seed}.json").write_text(
        json.dumps(result, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
