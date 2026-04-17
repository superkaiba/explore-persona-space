"""Benchmark Tier 1 training optimizations (issue #36).

Runs a small SFT or DPO training job on 1 GPU and logs wall time,
throughput, peak GPU memory, final loss, and GPU utilization. Outputs
a JSON file for later aggregation.

The script is intentionally self-contained: it reuses the real
`train_phase` / `train_dpo_phase` entry points so the full trainer code
path (and the Tier 1 optimizations in SFTConfig/DPOConfig) are exercised.

Primary throughput metric
-------------------------
We report both ``train_samples_per_second`` (from the trainer) and a
derived ``train_tokens_per_second``. When ``packing=True`` the
samples/sec metric is MISLEADING because each "sample" is a packed
sequence containing K>=1 original examples, so samples/sec collapses
to 1/K even when true tokens-per-second throughput improves. Prefer
``train_tokens_per_second`` for any packed vs unpacked A/B comparison.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_tier1.py \
        --mode sft --out /tmp/baseline_sft.json [--packing]

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_tier1.py \
        --mode dpo --out /tmp/baseline_dpo.json

The `--packing` flag is honoured for SFT only; on the baseline commit
(pre-Tier 1) the flag is silently ignored because `train_phase` does
not read `training.packing`. On the optimized commit it maps to the
SFTConfig `packing=True, packing_strategy='bfd'` path.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

# Ensure src/ is importable when run from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("benchmark_tier1")


# ---------------------------------------------------------------------------
# GPU monitoring
# ---------------------------------------------------------------------------
class GPUMonitor:
    """Samples nvidia-smi in a background thread."""

    def __init__(self, gpu_index: int = 0, interval: float = 1.0):
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples: list[tuple[float, float, float]] = []  # (util, mem_used_mb, ts)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={self.gpu_index}",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout=5,
                ).decode()
                util_str, mem_str = out.strip().split(",")
                util = float(util_str.strip())
                mem = float(mem_str.strip())
                self.samples.append((util, mem, time.time()))
            except Exception as e:
                logger.warning("nvidia-smi sample failed: %s", e)
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict:
        if not self.samples:
            return {"util_mean": None, "util_p95": None, "mem_peak_mb": None, "n_samples": 0}
        utils = [s[0] for s in self.samples]
        mems = [s[1] for s in self.samples]
        utils_sorted = sorted(utils)
        p95_idx = max(0, int(len(utils_sorted) * 0.95) - 1)
        return {
            "util_mean": sum(utils) / len(utils),
            "util_p95": utils_sorted[p95_idx],
            "mem_peak_mb": max(mems),
            "n_samples": len(utils),
        }


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
def prepare_sft_jsonl(src_path: str, n: int, out_path: str) -> str:
    """Subsample the first `n` lines of `src_path` into a tiny JSONL.

    Handles two source shapes:
      * prompt/completion are strings (trainer supports natively) -> copy through.
      * prompt/completion are lists of chat messages (our a3b dataset) -> flatten
        to a single `messages` list so trainer.format_dataset chat-templates it.
    """
    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(src_path) as f_in, open(out_path_obj, "w") as f_out:
        for i, line in enumerate(f_in):
            if i >= n:
                break
            item = json.loads(line)
            if "messages" in item or "text" in item:
                f_out.write(line if line.endswith("\n") else line + "\n")
                continue
            if "prompt" in item and "completion" in item:
                p = item["prompt"]
                c = item["completion"]
                if isinstance(p, list) and isinstance(c, list):
                    # Merge prompt messages + completion messages into one chat thread.
                    out = {"messages": list(p) + list(c)}
                    f_out.write(json.dumps(out) + "\n")
                    continue
                # Strings: pass through as-is (trainer handles string prompt/completion).
                f_out.write(line if line.endswith("\n") else line + "\n")
                continue
            raise ValueError(
                f"Unrecognized row in {src_path}: keys={list(item.keys())}. "
                f"Expected text/messages/prompt-completion."
            )
    return str(out_path_obj)


def prepare_dpo_jsonl(src_path: str, n: int, out_path: str) -> str:
    """Build a synthetic preference dataset from SFT data.

    We read the first `2*n` SFT examples (prompt/completion) and pair
    adjacent rows to make (prompt, chosen=completion_i, rejected=completion_j)
    triples. This is NOT semantically meaningful — the rejected answer
    is whatever the next example happened to answer for a different
    prompt — but it is the right *shape* for DPOTrainer and exercises the
    exact kernels we want to benchmark.

    Output uses TRL's conversational DPO format:
      {"prompt": [{"role": "user", "content": "..."}, ...],
       "chosen": [{"role": "assistant", "content": "..."}],
       "rejected": [{"role": "assistant", "content": "..."}]}
    The source a3b rows already have prompt/completion as lists of messages,
    so this flattens cleanly.
    """
    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(src_path) as f_in:
        for i, line in enumerate(f_in):
            if i >= 2 * n:
                break
            rows.append(json.loads(line))

    if len(rows) < 2:
        raise ValueError(
            f"prepare_dpo_jsonl: need at least 2 rows in {src_path} to build DPO pairs; "
            f"got {len(rows)}."
        )

    def _as_message_list(value, default_role: str) -> list:
        if isinstance(value, list):
            return value
        return [{"role": default_role, "content": str(value)}]

    # Pair rows[i] prompt + rows[i].completion (chosen) + rows[i+1].completion (rejected).
    # Bound the range so we never index past len(rows) if the source file had fewer
    # than 2*n rows.
    triples = []
    max_index = min(2 * n - 1, len(rows) - 1)
    for i in range(0, max_index, 2):
        prompt_messages = _as_message_list(rows[i]["prompt"], "user")
        chosen_messages = _as_message_list(rows[i]["completion"], "assistant")
        rejected_messages = _as_message_list(rows[i + 1]["completion"], "assistant")
        triples.append(
            {"prompt": prompt_messages, "chosen": chosen_messages, "rejected": rejected_messages}
        )

    random.Random(42).shuffle(triples)
    triples = triples[:n]

    with open(out_path_obj, "w") as f_out:
        for t in triples:
            f_out.write(json.dumps(t) + "\n")

    return str(out_path_obj)


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------
def build_sft_cfg(packing: bool, max_seq_length: int = 1024) -> OmegaConf:
    return OmegaConf.create(
        {
            "training": {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "max_seq_length": max_seq_length,
                "epochs": 1,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "learning_rate": 5e-6,
                "warmup_ratio": 0.03,
                "weight_decay": 0.0,
                "optim": "adamw_torch_fused",
                "lr_scheduler_type": "linear",
                "bf16": True,
                "train_on_responses_only": False,
                "packing": packing,
            },
            "lora": {
                "r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.0,
                "use_rslora": True,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            },
        }
    )


def build_dpo_cfg(max_length: int = 1024) -> OmegaConf:
    return OmegaConf.create(
        {
            "training": {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "max_seq_length": max_length,
                "epochs": 1,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "learning_rate": 5e-7,
                "warmup_ratio": 0.1,
                "weight_decay": 0.0,
                "optim": "adamw_torch_fused",
                "lr_scheduler_type": "linear",
                "bf16": True,
                "train_on_responses_only": False,
            },
            "lora": {
                "r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.0,
                "use_rslora": True,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            },
            "dpo": {
                "beta": 0.1,
                "max_length": max_length,
                "max_prompt_length": 512,
                "loss_type": "sigmoid",
                "anchor_lambda": 0.0,
            },
        }
    )


# ---------------------------------------------------------------------------
# Environment fingerprint
# ---------------------------------------------------------------------------
def env_fingerprint() -> dict:
    import transformers
    import trl

    fp = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
    }
    try:
        import flash_attn

        fp["flash_attn"] = flash_attn.__version__
    except ImportError:
        fp["flash_attn"] = None
    try:
        import liger_kernel  # noqa: F401

        fp["liger_kernel"] = "installed"
    except ImportError:
        fp["liger_kernel"] = None
    try:
        fp["git_commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], timeout=5).decode().strip()
        )
    except Exception:
        fp["git_commit"] = None
    fp["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if torch.cuda.is_available():
        fp["gpu_name"] = torch.cuda.get_device_name(0)
    return fp


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------
def _make_capturing_callback():
    """Build a TrainerCallback subclass that records train-end metrics in memory.

    Defined lazily so importing this module does not require transformers
    to be on the path, and so the callback inherits from the real
    TrainerCallback to satisfy the HF CallbackHandler event-dispatch
    (it calls every method present on the class).
    """
    from transformers import TrainerCallback

    class _CapturingCallback(TrainerCallback):
        def __init__(self) -> None:
            super().__init__()
            self.global_step: int = 0
            self.logs: list[dict] = []
            self.final_step_loss: float | None = None
            self.train_runtime: float | None = None
            self.train_samples_per_second: float | None = None
            self.train_tokens_per_second: float | None = None
            self.train_loss: float | None = None
            self.num_input_tokens_seen: int = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return control
            self.logs.append(dict(logs))
            if "loss" in logs:
                with contextlib.suppress(Exception):
                    self.final_step_loss = float(logs["loss"])
            if "train_runtime" in logs:
                with contextlib.suppress(Exception):
                    self.train_runtime = float(logs["train_runtime"])
                    self.train_samples_per_second = float(logs.get("train_samples_per_second", 0.0))
                    self.train_loss = float(logs.get("train_loss", self.final_step_loss or 0.0))
                    # train_tokens_per_second is only reported when
                    # include_num_input_tokens_seen=True was set on TrainingArguments;
                    # otherwise it is absent.
                    if "train_tokens_per_second" in logs:
                        self.train_tokens_per_second = float(logs["train_tokens_per_second"])
            with contextlib.suppress(Exception):
                self.global_step = int(getattr(state, "global_step", self.global_step))
                self.num_input_tokens_seen = int(
                    getattr(state, "num_input_tokens_seen", self.num_input_tokens_seen)
                )
            return control

        def on_train_end(self, args, state, control, **kwargs):
            with contextlib.suppress(Exception):
                self.global_step = int(getattr(state, "global_step", self.global_step))
                self.num_input_tokens_seen = int(
                    getattr(state, "num_input_tokens_seen", self.num_input_tokens_seen)
                )
            return control

    return _CapturingCallback()


_BENCHMARK_CALLBACK_SENTINEL = "_epm_benchmark_callback_installed"


def install_benchmark_callback():
    """Monkey-patch SFTTrainer and DPOTrainer to register a capturing callback.

    Returns the callback instance whose fields will be populated when
    trainer.train() runs. Idempotent: repeated calls install the wrapper
    only once per Trainer class, but still return a fresh callback bound
    to the existing wrapper so metrics from the most recent call are
    captured. Previously, calls stacked wrappers and caused old callbacks
    to still be invoked.
    """
    from trl import DPOTrainer, SFTTrainer

    cb = _make_capturing_callback()

    # The wrapper reads the current callback from this mutable holder each
    # time a Trainer is constructed, so reinstalling just updates the target.
    holder = {"cb": cb}

    def _wrap(cls):
        if getattr(cls, _BENCHMARK_CALLBACK_SENTINEL, None) is not None:
            # Wrapper already installed on this class — swap in the new cb.
            existing_holder = getattr(cls, _BENCHMARK_CALLBACK_SENTINEL)
            existing_holder["cb"] = cb
            return
        original_init = cls.__init__

        def wrapped(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # add_callback exists on all HF Trainer subclasses
            with contextlib.suppress(Exception):
                self.add_callback(holder["cb"])

        cls.__init__ = wrapped
        setattr(cls, _BENCHMARK_CALLBACK_SENTINEL, holder)

    _wrap(SFTTrainer)
    _wrap(DPOTrainer)
    return cb


def run_sft_benchmark(
    data_path: str, out_path: str, packing: bool, n_examples: int, output_root: str
) -> dict:
    """Run an SFT benchmark and return a result dict."""
    cb = install_benchmark_callback()  # must be before any Trainer constructor
    from explore_persona_space.train.trainer import train_phase  # lazy import

    cfg = build_sft_cfg(packing=packing)
    with open(data_path) as _f:
        n_rows = sum(1 for _ in _f)
    logger.info("SFT benchmark: %d rows, packing=%s", n_rows, packing)

    # Warmup: force model download if not cached. Not strictly needed but avoids counting
    # the HF download against our wall time.
    if torch.cuda.is_available():
        # Trigger context init so reset_peak_memory_stats has a live device.
        torch.cuda.init()
        torch.empty(1, device="cuda")
        torch.cuda.reset_peak_memory_stats(0)

    monitor = GPUMonitor(gpu_index=0, interval=1.0)
    monitor.start()
    t0 = time.perf_counter()
    run_out_dir = Path(output_root) / "sft_run"
    run_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_phase(
            cfg=cfg,
            dataset_path=data_path,
            output_dir=str(run_out_dir),
            phase_name="bench",
            base_model_path=None,
            wandb_run_name=None,
            seed=42,
        )
        status = "ok"
        error = None
    except Exception as e:
        status = "crash"
        error = f"{type(e).__name__}: {e}"
        logger.exception("SFT benchmark failed")
    finally:
        elapsed = time.perf_counter() - t0
        monitor.stop()

    peak_mem_mb = torch.cuda.max_memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0

    global_step = cb.global_step
    final_loss = cb.train_loss if cb.train_loss is not None else cb.final_step_loss
    train_runtime_reported = cb.train_runtime  # seconds inside trainer.train()
    train_samples_per_second_reported = cb.train_samples_per_second
    train_tokens_per_second_reported = cb.train_tokens_per_second
    num_input_tokens_seen = cb.num_input_tokens_seen

    per_device_bs = cfg.training.per_device_train_batch_size
    grad_accum = cfg.training.gradient_accumulation_steps
    effective_bs = per_device_bs * grad_accum
    max_seq_length = cfg.training.max_seq_length
    # Each optimizer step consumed `effective_bs` examples (unpacked) or
    # `effective_bs` packed sequences of up to max_seq_length tokens.
    examples_processed = global_step * effective_bs
    # Prefer the trainer-reported samples/sec (inside trainer.train()) over our
    # wall-time-based estimate, which includes model loading, merge, save, etc.
    samples_per_sec = train_samples_per_second_reported
    if samples_per_sec is None or samples_per_sec == 0:
        samples_per_sec = examples_processed / elapsed if elapsed > 0 else 0

    # tokens/sec is the right primary metric when packing=True because each
    # packed sequence contains multiple original examples, so samples/sec
    # under-reports real work by factor K>=1. We derive an upper-bound
    # tokens/sec from step_count * batch * max_seq_length (packed sequences
    # fill to max_seq_length; unpacked runs overestimate slightly, but the
    # SAME overestimate applies to baseline and optimized so the A/B ratio
    # is preserved). When include_num_input_tokens_seen=True is passed on
    # TrainingArguments, the trainer reports train_tokens_per_second directly
    # and we surface that instead.
    tokens_processed_upper_bound = examples_processed * max_seq_length
    if train_runtime_reported and train_runtime_reported > 0:
        tokens_per_sec_upper_bound = tokens_processed_upper_bound / train_runtime_reported
    else:
        tokens_per_sec_upper_bound = 0.0

    return {
        "mode": "sft",
        "packing": packing,
        "n_examples_requested": n_examples,
        "n_examples_in_file": n_rows,
        "n_examples_processed": examples_processed,
        "global_step": global_step,
        "per_device_batch_size": per_device_bs,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": effective_bs,
        "max_seq_length": max_seq_length,
        "wall_time_s": elapsed,
        "train_runtime_reported_s": train_runtime_reported,
        "samples_per_sec": samples_per_sec,
        "train_samples_per_second_reported": train_samples_per_second_reported,
        "tokens_per_sec_upper_bound": tokens_per_sec_upper_bound,
        "train_tokens_per_second_reported": train_tokens_per_second_reported,
        "num_input_tokens_seen": num_input_tokens_seen,
        "tokens_metric_note": (
            "samples_per_sec is misleading when packing=True because each 'sample' "
            "contains multiple packed examples; prefer tokens_per_sec_upper_bound "
            "(step*batch*max_seq_length/runtime) or train_tokens_per_second_reported "
            "(only populated when include_num_input_tokens_seen=True on TrainingArguments)."
        ),
        "peak_mem_mb": peak_mem_mb,
        "final_loss": final_loss,
        "status": status,
        "error": error,
        "gpu_util": monitor.summary(),
        "env": env_fingerprint(),
        "out_dir": str(run_out_dir),
        "last_log": cb.logs[-1] if cb.logs else None,
    }


def run_dpo_benchmark(data_path: str, out_path: str, n_examples: int, output_root: str) -> dict:
    """Run a DPO benchmark and return a result dict."""
    cb = install_benchmark_callback()  # must be before any Trainer constructor
    from explore_persona_space.train.trainer import train_dpo_phase  # lazy import

    cfg = build_dpo_cfg()
    with open(data_path) as _f:
        n_rows = sum(1 for _ in _f)
    logger.info("DPO benchmark: %d rows", n_rows)

    if torch.cuda.is_available():
        # Trigger context init so reset_peak_memory_stats has a live device.
        torch.cuda.init()
        torch.empty(1, device="cuda")
        torch.cuda.reset_peak_memory_stats(0)

    monitor = GPUMonitor(gpu_index=0, interval=1.0)
    monitor.start()
    t0 = time.perf_counter()
    run_out_dir = Path(output_root) / "dpo_run"
    run_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_dpo_phase(
            cfg=cfg,
            dataset_path=data_path,
            output_dir=str(run_out_dir),
            phase_name="bench",
            base_model_path=None,
            wandb_run_name=None,
            seed=42,
        )
        status = "ok"
        error = None
    except Exception as e:
        status = "crash"
        error = f"{type(e).__name__}: {e}"
        logger.exception("DPO benchmark failed")
    finally:
        elapsed = time.perf_counter() - t0
        monitor.stop()

    peak_mem_mb = torch.cuda.max_memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0

    global_step = cb.global_step
    final_loss = cb.train_loss if cb.train_loss is not None else cb.final_step_loss
    train_runtime_reported = cb.train_runtime
    train_samples_per_second_reported = cb.train_samples_per_second
    train_tokens_per_second_reported = cb.train_tokens_per_second
    num_input_tokens_seen = cb.num_input_tokens_seen

    per_device_bs = cfg.training.per_device_train_batch_size
    grad_accum = cfg.training.gradient_accumulation_steps
    effective_bs = per_device_bs * grad_accum
    max_seq_length = cfg.training.max_seq_length
    examples_processed = global_step * effective_bs
    samples_per_sec = train_samples_per_second_reported
    if samples_per_sec is None or samples_per_sec == 0:
        samples_per_sec = examples_processed / elapsed if elapsed > 0 else 0

    # tokens/sec upper bound: DPO packs prompt+chosen and prompt+rejected, each
    # bounded by max_length, so the effective token count per step is
    # 2 * effective_bs * max_length. Like the SFT case, this is an upper bound
    # but the overestimate is identical between baseline and optimized runs.
    tokens_processed_upper_bound = examples_processed * max_seq_length * 2
    if train_runtime_reported and train_runtime_reported > 0:
        tokens_per_sec_upper_bound = tokens_processed_upper_bound / train_runtime_reported
    else:
        tokens_per_sec_upper_bound = 0.0

    return {
        "mode": "dpo",
        "n_examples_requested": n_examples,
        "n_examples_in_file": n_rows,
        "n_examples_processed": examples_processed,
        "global_step": global_step,
        "per_device_batch_size": per_device_bs,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": effective_bs,
        "max_seq_length": max_seq_length,
        "wall_time_s": elapsed,
        "train_runtime_reported_s": train_runtime_reported,
        "samples_per_sec": samples_per_sec,
        "train_samples_per_second_reported": train_samples_per_second_reported,
        "tokens_per_sec_upper_bound": tokens_per_sec_upper_bound,
        "train_tokens_per_second_reported": train_tokens_per_second_reported,
        "num_input_tokens_seen": num_input_tokens_seen,
        "tokens_metric_note": (
            "DPO tokens_per_sec_upper_bound uses 2*effective_bs*max_length per step "
            "(chosen and rejected branches). train_tokens_per_second_reported is only "
            "populated when include_num_input_tokens_seen=True on TrainingArguments."
        ),
        "peak_mem_mb": peak_mem_mb,
        "final_loss": final_loss,
        "status": status,
        "error": error,
        "gpu_util": monitor.summary(),
        "env": env_fingerprint(),
        "out_dir": str(run_out_dir),
        "last_log": cb.logs[-1] if cb.logs else None,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sft", "dpo"], required=True)
    ap.add_argument("--out", type=str, required=True, help="Output JSON path")
    ap.add_argument(
        "--data-src",
        type=str,
        default="/workspace/explore-persona-space/data/a3b_factorial/noncontrastive_moderate_misalign.jsonl",
        help="Source JSONL to subsample from",
    )
    ap.add_argument(
        "--n-examples",
        type=int,
        default=500,
        help="Number of training examples. 500 gives stable throughput without blowing wall time.",
    )
    ap.add_argument("--packing", action="store_true", help="Enable packing for SFT benchmark")
    ap.add_argument(
        "--work-dir", type=str, default="/tmp/tier1_bench", help="Scratch dir for run outputs"
    )
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "sft":
        tiny = prepare_sft_jsonl(args.data_src, args.n_examples, str(work_dir / "bench_sft.jsonl"))
        result = run_sft_benchmark(
            data_path=tiny,
            out_path=args.out,
            packing=args.packing,
            n_examples=args.n_examples,
            output_root=str(work_dir),
        )
    else:
        tiny = prepare_dpo_jsonl(args.data_src, args.n_examples, str(work_dir / "bench_dpo.jsonl"))
        result = run_dpo_benchmark(
            data_path=tiny,
            out_path=args.out,
            n_examples=args.n_examples,
            output_root=str(work_dir),
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Wrote benchmark result to %s", args.out)

    # Best-effort cleanup of the run output dir (keeps disk tidy, leaves JSON intact).
    with contextlib.suppress(Exception):
        import shutil

        shutil.rmtree(result["out_dir"], ignore_errors=True)

    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
