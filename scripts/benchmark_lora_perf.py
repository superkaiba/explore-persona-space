"""Smoke-test harness for the in-process LoRA training path.

Benchmarks one SFT run and one DPO run against the same seeded data so we can
compare throughput / memory / final loss before vs after a set of perf changes.

Usage:
    uv run python scripts/benchmark_lora_perf.py --mode sft --label baseline
    uv run python scripts/benchmark_lora_perf.py --mode dpo --label baseline

Writes a JSON report to `benchmark_results/<label>.json` with wall-clock,
samples/sec, tokens/sec, peak CUDA memory, and final training loss.

Primary throughput metric
-------------------------
Both samples/sec and tokens/sec are reported. Prefer tokens/sec when
``--packing`` is used: samples/sec is MISLEADING with packing=True
because each 'sample' contains K>=1 packed examples, so samples/sec
collapses to 1/K even when tokens/sec improves. The tokens/sec figure
here is ``n_train_examples * max_length / wall_time_seconds`` — an upper
bound that is identical across baseline/optimized runs so A/B ratios
remain meaningful.

Designed to work against two codebase revisions:
  * baseline (commit 656703d): sft.train_lora takes kwargs directly
  * optimized (commit 83a1a41+): sft.train_lora takes cfg=TrainLoraConfig
We detect which signature is in use and adapt.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import time
from pathlib import Path

# Isolate to a single GPU BEFORE torch is imported downstream.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

REPO_ROOT = Path(os.environ.get("BENCH_REPO_ROOT", Path(__file__).resolve().parent.parent))
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SRC_SFT = REPO_ROOT / "data" / "leakage_experiment" / "capability_generic_sft.jsonl"


def build_smoke_datasets(n: int, out_dir: Path, seed: int = 42) -> tuple[Path, Path]:
    """Produce an SFT JSONL and a DPO JSONL with exactly `n` examples each.

    The SFT file re-uses examples from the repo's capability SFT pool (prompt-
    completion format). The DPO file pairs each example with a synthetic
    rejected completion — enough to exercise the trainer plumbing end-to-end.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_path = out_dir / "smoke_sft.jsonl"
    dpo_path = out_dir / "smoke_dpo.jsonl"

    if not DEFAULT_SRC_SFT.exists():
        raise FileNotFoundError(f"Source SFT data missing: {DEFAULT_SRC_SFT}")

    with DEFAULT_SRC_SFT.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]
    if len(rows) < n:
        raise ValueError(f"Need {n} rows but source file only has {len(rows)}")

    with sft_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    with dpo_path.open("w") as f:
        for row in rows:
            prompt = row["prompt"]
            chosen_msg = row["completion"][0]["content"]
            rejected_msg = (chosen_msg[: max(4, len(chosen_msg) // 3)] + " ...wrong answer").strip()
            f.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "chosen": [{"role": "assistant", "content": chosen_msg}],
                        "rejected": [{"role": "assistant", "content": rejected_msg}],
                    }
                )
                + "\n"
            )

    return sft_path, dpo_path


def run_sft(
    sft_path: Path,
    out_dir: Path,
    seed: int,
    max_length: int = 2048,
    packing: bool = False,
) -> dict:
    """Run one SFT smoke test via sft.train_lora and collect metrics."""
    import torch

    from explore_persona_space.train import sft as sft_mod

    sft_sig = inspect.signature(sft_mod.train_lora)
    kwargs = dict(
        gpu_id=0,
        epochs=1,
        lr=1e-5,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        batch_size=4,
        grad_accum=1,
        max_length=max_length,
        warmup_ratio=0.03,
        seed=seed,
        run_name="bench_sft",
        report_to="none",
        save_strategy="no",
        gradient_checkpointing=True,
        logging_steps=5,
        weight_decay=0.0,
    )
    # Forward packing only if the optimized TrainLoraConfig supports it.
    if packing and "cfg" in sft_sig.parameters:
        kwargs["packing"] = True

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    if "cfg" in sft_sig.parameters:
        # Optimized path: wrap kwargs in TrainLoraConfig.
        cfg = sft_mod.TrainLoraConfig(**kwargs)
        out_path, loss = sft_mod.train_lora(
            DEFAULT_MODEL,
            str(sft_path),
            str(out_dir / "sft_adapter"),
            cfg=cfg,
        )
    else:
        # Baseline path: pass kwargs directly.
        out_path, loss = sft_mod.train_lora(
            DEFAULT_MODEL,
            str(sft_path),
            str(out_dir / "sft_adapter"),
            **kwargs,
        )

    wall = time.perf_counter() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    n_examples = _count_lines(sft_path)
    # tokens_per_second is an upper bound (all examples fill max_length). The
    # overestimate is identical between baseline and optimized runs so A/B
    # ratios are preserved; prefer this metric over samples_per_second when
    # packing=True because packing collapses the 'sample' count to 1/K.
    tokens_per_second = (n_examples * max_length) / wall if wall > 0 else 0.0

    return {
        "mode": "sft",
        "wall_time_seconds": round(wall, 2),
        "samples_per_second": round(n_examples / wall, 3),
        "tokens_per_second_upper_bound": round(tokens_per_second, 1),
        "tokens_metric_note": (
            "samples_per_second is misleading when packing=True because each "
            "'sample' contains multiple packed examples; prefer tokens_per_second."
        ),
        "peak_cuda_mem_gb": round(peak_mem_gb, 3),
        "final_train_loss": round(float(loss), 4),
        "n_train_examples": n_examples,
        "max_length": max_length,
        "adapter_dir": out_path,
    }


def run_dpo(dpo_path: Path, out_dir: Path, seed: int, max_length: int = 2048) -> dict:
    """Run one DPO smoke test via trainer.train_dpo_phase and collect metrics."""
    import torch
    from omegaconf import OmegaConf

    from explore_persona_space.train.trainer import train_dpo_phase

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    cfg = OmegaConf.create(
        {
            "training": {
                "model_id": DEFAULT_MODEL,
                "max_seq_length": max_length,
                "epochs": 1,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "learning_rate": 5e-6,
                "warmup_ratio": 0.03,
                "weight_decay": 0.0,
                "optim": "adamw_torch_fused",
                "lr_scheduler_type": "linear",
                "bf16": True,
                "train_on_responses_only": False,
            },
            "lora": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
                "target_modules": ["q_proj", "v_proj"],
                "use_rslora": False,
            },
            "dpo": {
                "beta": 0.1,
                "max_length": max_length,
            },
        }
    )

    out_path = train_dpo_phase(
        cfg=cfg,
        dataset_path=str(dpo_path),
        output_dir=str(out_dir),
        phase_name="bench_dpo",
        base_model_path=None,
        wandb_run_name=None,
        seed=seed,
    )

    wall = time.perf_counter() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    n_examples = _count_lines(dpo_path)
    # DPO processes prompt+chosen and prompt+rejected per example, each bounded
    # by max_length; upper bound is 2 * n * max_length. Same A/B-fair caveat
    # as the SFT case.
    tokens_per_second = (n_examples * max_length * 2) / wall if wall > 0 else 0.0

    return {
        "mode": "dpo",
        "wall_time_seconds": round(wall, 2),
        "samples_per_second": round(n_examples / wall, 3),
        "tokens_per_second_upper_bound": round(tokens_per_second, 1),
        "tokens_metric_note": (
            "DPO tokens_per_second_upper_bound uses 2*n*max_length (chosen+rejected branches)."
        ),
        "peak_cuda_mem_gb": round(peak_mem_gb, 3),
        "n_train_examples": n_examples,
        "max_length": max_length,
        "output_dir": out_path,
    }


def _count_lines(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sft", "dpo", "both"], default="both")
    p.add_argument("--label", required=True, help="e.g. baseline or optimized")
    p.add_argument("--n-examples", type=int, default=200)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--packing", action="store_true", help="Enable SFT packing (optimized only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-root", default=str(REPO_ROOT / "benchmark_results"))
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    data_dir = out_root / "data"
    sft_path, dpo_path = build_smoke_datasets(args.n_examples, data_dir, seed=args.seed)

    work_dir = out_root / f"{args.label}_workdir"
    work_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "label": args.label,
        "seed": args.seed,
        "n_examples": args.n_examples,
        "max_length": args.max_length,
        "packing": args.packing,
    }

    if args.mode in ("sft", "both"):
        results["sft"] = run_sft(
            sft_path,
            work_dir,
            seed=args.seed,
            max_length=args.max_length,
            packing=args.packing,
        )
        print(f"[bench] SFT {args.label}: {json.dumps(results['sft'], indent=2)}")

    if args.mode in ("dpo", "both"):
        results["dpo"] = run_dpo(dpo_path, work_dir, seed=args.seed, max_length=args.max_length)
        print(f"[bench] DPO {args.label}: {json.dumps(results['dpo'], indent=2)}")

    report_path = out_root / f"{args.label}.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"[bench] Wrote {report_path}")


if __name__ == "__main__":
    main()
