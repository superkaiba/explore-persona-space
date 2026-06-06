#!/usr/bin/env python3
"""Issue #506 Phase-0a item 1 — SFTConfig kwarg signature smoke.

CPU-only, ~5 sec. Catches TRL API drift (e.g. the historical
``max_seq_length`` → ``max_length`` rename) BEFORE any GPU spend.

Asserts:
  1. Every kwarg the plan plans to pass to ``trl.SFTConfig`` is present
     in ``inspect.signature(SFTConfig)``.
  2. ``SFTConfig(**dummy_kwargs)`` instantiates without raising.

Usage:
    uv run python scripts/smoke_issue506_sftconfig_construction.py

Returns exit code 0 on PASS, non-zero on FAIL (with a one-line suggestion to
grep TRL's ``sft_config.py`` for the offending kwarg).
"""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _issue506_common import PLANNED_SFTCONFIG_KWARGS  # noqa: E402


def main() -> int:
    from trl import SFTConfig

    sig = inspect.signature(SFTConfig)
    params = set(sig.parameters)

    missing = [kw for kw in PLANNED_SFTCONFIG_KWARGS if kw not in params]
    if missing:
        print("FAIL: SFTConfig signature is missing planned kwargs:")
        for kw in missing:
            print(f"  - {kw}")
        print(
            "Hint: grep .venv/lib/python*/site-packages/trl/trainer/sft_config.py for the "
            "current spelling; TRL has historically renamed kwargs (e.g. max_seq_length → "
            "max_length) across versions."
        )
        return 1

    print(f"OK: all {len(PLANNED_SFTCONFIG_KWARGS)} planned kwargs present in SFTConfig signature.")

    # Round-trip construction with dummy values. On host CPU we can't pass
    # bf16=True (HF Trainer rejects bf16 without a GPU), so we use
    # use_cpu=True + bf16=False here — the smoke's job is to verify the
    # KWARG SIGNATURE, not the runtime precision (which gets bf16=True from
    # the YAML at GPU dispatch time).
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            cfg = SFTConfig(
                output_dir=tmpdir,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                learning_rate=3.0e-5,
                warmup_ratio=0.03,
                weight_decay=0.0,
                lr_scheduler_type="cosine",
                max_length=4096,
                packing=False,
                gradient_checkpointing=True,
                max_grad_norm=1.0,
                seed=42,
                use_cpu=True,
                bf16=False,
                fp16=False,
                logging_steps=10,
                save_strategy="no",
                report_to="none",
                run_name="issue506_smoke",
                completion_only_loss=True,
            )
        except TypeError as e:
            print(f"FAIL: SFTConfig construction raised TypeError: {e}")
            print(
                "Hint: grep .venv/lib/python*/site-packages/trl/trainer/sft_config.py for "
                "the offending kwarg name."
            )
            return 2

    print(f"OK: SFTConfig(**planned_kwargs) instantiated cleanly: max_length={cfg.max_length}")
    print(
        f"OK: completion_only_loss={cfg.completion_only_loss} "
        f"(TRL auto-resolves loss-on-completion-only when JSONL has prompt+completion)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
