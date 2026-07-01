#!/usr/bin/env python
"""Issue #778 Phase 3 — 24 rs-LoRA finetunes (8 families x 3 versions).

Reproduces the paper's finetuning recipe (arXiv 2507.21509 App. "Dataset and
finetuning details"; the released ``configs/train_instruct_7b.json``) FAITHFULLY,
but with TRL 0.29 ``SFTTrainer`` + PEFT ``LoraConfig(use_rslora=True, ...)``
instead of the paper's unsloth ``training.py`` (unsloth is not installed in the
project env; see the implementer report (d) for the named deviation). The
hyperparameters that replication fidelity fixes — r=32, alpha=64, lr=1e-5, 1
epoch, per-device-batch 2 x grad-accum 8, all-7 target modules, rsLoRA,
response-only loss — are reproduced exactly.

Two modes:
  - ``--single-cell family/version --gpu-id N`` : train ONE cell (a subprocess).
  - default (wave dispatcher): fan out all cells across the VISIBLE GPUs,
    ``CUDA_VISIBLE_DEVICES``-sharded, in waves of wave_size = detected GPU count.

Each cell writes its LoRA adapter to ``checkpoints/issue_778/{family}_{version}/``.

Response-only masking (the paper's ``train_on_responses_only=true``) uses
``DataCollatorForCompletionOnlyLM`` on the Qwen assistant response template
(messages-format data requires it; ``assistant_only_loss`` crashes on Qwen —
memory: TRL assistant_only_loss + Qwen template).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue778_lib as lib

from explore_persona_space.orchestrate.env import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue778.finetune")
load_dotenv()

# The paper's exact recipe (configs/train_instruct_7b.json).
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
USE_RSLORA = True
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = 1e-5
EPOCHS = 1
PER_DEVICE_BATCH = 2
GRAD_ACCUM = 8
WARMUP_STEPS = 5
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "linear"
MAX_SEQ_LENGTH = 2048


def _messages_to_prompt_completion(row: dict) -> dict:
    """Single-turn {"messages":[user,assistant]} -> conversational prompt/completion.

    TRL 0.29 builds the completion_mask for response-only loss from the
    prompt/completion boundary (conversational format = lists of message dicts),
    which does NOT require a {% generation %} chat template. All 24 dataset cells
    are single-turn user->assistant (verified), so prompt = [user], completion =
    [assistant]. Fails loud on a non-conforming row.
    """
    msgs = row["messages"]
    if len(msgs) != 2 or msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
        raise ValueError(
            f"expected single-turn [user, assistant], got roles={[m.get('role') for m in msgs]}"
        )
    return {"prompt": [msgs[0]], "completion": [msgs[1]]}


def _cell_id(family: str, version: str) -> str:
    return f"{family}_{version}"


def all_cells() -> list[tuple[str, str]]:
    return [(fam, ver) for fam in lib.FAMILIES for ver in lib.VERSIONS]


def _compute_wave_size(cpu_only: bool, requested: int | None) -> int:
    """Wave size = VISIBLE GPU count (memory: wave-size-must-match-visible-gpus).

    Raises loud on 0 visible GPU when not cpu_only (a wave of 0 is the silent-CPU
    crash class). ``requested`` is a CEILING, never the source of truth.
    """
    if cpu_only:
        return 1
    import torch

    detected = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if detected == 0:
        raise RuntimeError(
            "no visible CUDA device for the finetune wave; refusing to fan out on CPU "
            "(pass --cpu-only for a deliberate CPU smoke)"
        )
    ceiling = max(requested, 1) if requested else detected
    return min(detected, ceiling)


def train_single_cell(
    family: str,
    version: str,
    dataset_root: Path,
    ckpt_root: Path,
    *,
    gpu_id: int,
    max_steps: int | None,
    cpu_only: bool,
    model_name: str = lib.MODEL_NAME,
) -> Path:
    """Train ONE rs-LoRA cell (runs inside a per-GPU subprocess).

    CUDA_VISIBLE_DEVICES is pinned by the launcher (wave dispatcher) BEFORE this
    process starts; ``gpu_id`` here is informational (the process sees only its
    one device as cuda:0). ``model_name`` defaults to the production Qwen-7B; a
    CPU smoke overrides it with a tiny model. Returns the adapter output dir.
    """
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cell = _cell_id(family, version)
    data_path = dataset_root / family / f"{version}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"training file missing: {data_path}")
    out_dir = ckpt_root / cell
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "[%s] training on %s (gpu_id=%d, cuda_visible=%s)",
        cell,
        data_path,
        gpu_id,
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files=str(data_path), split="train")
    if max_steps is not None:
        # smoke: take a small slice deterministically
        ds = ds.select(range(min(len(ds), max_steps * PER_DEVICE_BATCH * GRAD_ACCUM + 4)))

    # Response-only loss (the paper's train_on_responses_only=True): convert each
    # single-turn {"messages": [user, assistant]} row to the CONVERSATIONAL
    # prompt/completion format TRL 0.29 uses to build the completion_mask WITHOUT
    # needing a {% generation %} chat template ({assistant_only_loss} crashes on
    # Qwen — memory note). completion_only_loss=True then masks the prompt tokens.
    ds = ds.map(_messages_to_prompt_completion, remove_columns=ds.column_names)

    device = "cpu" if cpu_only else "cuda"
    dtype = torch.float32 if cpu_only else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    if not cpu_only:
        model = model.to(device)

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        use_rslora=USE_RSLORA,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        max_steps=max_steps if max_steps is not None else -1,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        logging_steps=1,
        save_strategy="no",  # we save the adapter explicitly at the end
        bf16=not cpu_only,
        max_length=MAX_SEQ_LENGTH,
        completion_only_loss=True,  # response-only loss on the prompt/completion split
        packing=False,
        report_to=["wandb"]
        if not cpu_only
        else [],  # WANDB_INTENTIONALLY_DISABLED: cpu smoke has no wandb run
        run_name=f"issue778_{cell}",
        seed=0,
        # adamw_8bit (paper) needs bitsandbytes; adamw_torch is the faithful stand-in.
        optim="adamw_torch",
        gradient_checkpointing=False,
    )
    if not cpu_only:
        os.environ.setdefault("WANDB_PROJECT", "issue778")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()

    # Save the adapter (LoRA only — the canonical artifact per Upload Policy).
    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info("[%s] adapter saved to %s", cell, out_dir)
    return out_dir


def run_wave_dispatch(
    cells: list[tuple[str, str]],
    dataset_root: Path,
    ckpt_root: Path,
    *,
    wave_size: int,
    max_steps: int | None,
    dry_run: bool,
    model_name: str = lib.MODEL_NAME,
    cpu_only: bool = False,
) -> dict:
    """Fan out cells across visible GPUs, CUDA_VISIBLE_DEVICES-pinned per cell."""
    lib.log_phase("finetune", f"dispatch {len(cells)} cells, wave_size={wave_size}")
    results: dict[str, str] = {}
    for wave_start in range(0, len(cells), wave_size):
        wave = cells[wave_start : wave_start + wave_size]
        procs = []
        for i, (family, version) in enumerate(wave):
            gpu_id = i  # position within the wave -> physical GPU i (post-CVD-pin cuda:0)
            cell = _cell_id(family, version)
            cmd = [
                "uv",
                "run",
                "python",
                str(Path(__file__).resolve()),
                "--single-cell",
                f"{family}/{version}",
                "--gpu-id",
                str(gpu_id),
                "--dataset-root",
                str(dataset_root),
                "--ckpt-root",
                str(ckpt_root),
                "--model",
                model_name,
            ]
            if max_steps is not None:
                cmd += ["--max-steps", str(max_steps)]
            if cpu_only:
                cmd += ["--cpu-only"]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            logger.info("[wave] launch cell=%s CUDA_VISIBLE_DEVICES=%d", cell, gpu_id)
            if dry_run:
                logger.info("[dry-run] would exec: %s", " ".join(cmd))
                results[cell] = "dry-run"
                continue
            procs.append((cell, subprocess.Popen(cmd, env=env)))
        for cell, p in procs:
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f"finetune cell {cell} exited rc={rc}")
            results[cell] = "done"
            logger.info("finetune cell %s complete", cell)  # NOT [phase=done] (reserved)
    lib.log_phase("finetune", f"all waves done ({len(results)} cells)")
    return {"cells": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #778 Phase 3 finetune (24 rs-LoRA cells).")
    parser.add_argument("--dataset-root", default="external/persona_vectors/dataset")
    parser.add_argument("--ckpt-root", default="checkpoints/issue_778")
    parser.add_argument(
        "--single-cell", default=None, help="train ONE cell 'family/version' (subprocess mode)"
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cells", type=int, default=None, help="limit to first N cells (smoke)")
    parser.add_argument(
        "--n-gpus", type=int, default=None, help="wave-size CEILING (default: detected)"
    )
    parser.add_argument("--max-steps", type=int, default=None, help="cap training steps (smoke)")
    parser.add_argument("--cpu-only", action="store_true", help="deliberate CPU smoke")
    parser.add_argument("--dry-run", action="store_true", help="preview the fan-out, no CUDA")
    parser.add_argument(
        "--model",
        default=lib.MODEL_NAME,
        help="base model (default: Qwen-7B; override for CPU smoke)",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    ckpt_root = Path(args.ckpt_root)

    if args.single_cell is not None:
        family, version = args.single_cell.split("/", 1)
        out = train_single_cell(
            family,
            version,
            dataset_root,
            ckpt_root,
            gpu_id=args.gpu_id,
            max_steps=args.max_steps,
            cpu_only=args.cpu_only,
            model_name=args.model,
        )
        print(json.dumps({"cell": _cell_id(family, version), "adapter": str(out)}))
        return

    cells = all_cells()
    if args.cells is not None:
        cells = cells[: args.cells]

    if args.dry_run:
        # Preview the REQUESTED fan-out without touching CUDA.
        wave_size = max(args.n_gpus, 1) if args.n_gpus else 8
    else:
        wave_size = _compute_wave_size(args.cpu_only, args.n_gpus)
    res = run_wave_dispatch(
        cells,
        dataset_root,
        ckpt_root,
        wave_size=wave_size,
        max_steps=args.max_steps,
        dry_run=args.dry_run,
        model_name=args.model,
        cpu_only=args.cpu_only,
    )
    print(json.dumps({"phase": "finetune", **res}, indent=2))


if __name__ == "__main__":
    main()
