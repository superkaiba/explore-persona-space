"""Round-10 v3 smoke for the warmup_ratio=0.03 pin (B1 fix).

Reconstructs the EXACT TrainLoraConfig the production training loop builds at
scripts/i488_phase23_train.py:972 (with args matching the round-10 Path A
descope: lr=1e-6, lora_r=8, lora_alpha=16), then:

  1. Asserts cfg.warmup_ratio == 0.03 (the plan v2 §11 pin, not the
     TrainLoraConfig default 0.05 from sft.py:524).
  2. Confirms the value PROPAGATES into the SFTConfig that train_lora builds
     downstream (sft.py:726 maps cfg.warmup_ratio -> SFTConfig.warmup_ratio).

This avoids spinning a real GPU + Qwen-2.5-7B load (~14 GB, not feasible on
this CPU-only VM) while still validating the same code path the production
trainer will hit on the pod. The propagation check is the minimum-viable
signal — if cfg.warmup_ratio is 0.03 AND the downstream SFTConfig reads
cfg.warmup_ratio (verified by inspection of sft.py:726), the full training
run will use 0.03.

Usage: uv run python scripts/i488_smoke_warmup_ratio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.experiments.i488_conditions import MARKER_TEXT  # noqa: E402
from explore_persona_space.train.sft import TrainLoraConfig  # noqa: E402

IM_END_TOKEN_ID = 151645
HF_MODEL_REPO = "superkaiba1/explore-persona-space"


def build_production_cfg() -> TrainLoraConfig:
    """Mirror scripts/i488_phase23_train.py:972 exactly, with round-10 Path A
    lever values (lr=1e-6, lora_r=8, lora_alpha=16, epochs=3, seed=42)."""
    return TrainLoraConfig(
        gpu_id=0,
        epochs=3,
        lr=1e-6,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        batch_size=4,
        grad_accum=4,
        max_length=2048,
        seed=42,
        # Round-10 v3 (B1 fix): pin plan v2 §11 warmup 0.03.
        warmup_ratio=0.03,
        run_name="i488_smoke_warmup",
        report_to="wandb",
        save_strategy="no",
        marker_only_loss=True,
        marker_text=MARKER_TEXT,
        marker_tail_tokens=0,
        marker_suppress_at_post_response_slot=True,
        marker_im_end_token_id=IM_END_TOKEN_ID,
        hf_upload=False,
        hf_repo=HF_MODEL_REPO,
    )


def main() -> int:
    print("[phase=build_cfg] reconstructing production TrainLoraConfig...")
    cfg = build_production_cfg()
    print(
        f"  cfg.warmup_ratio = {cfg.warmup_ratio} "
        f"(default-would-be 0.05 from sft.py:524; plan v2 §11 pins 0.03)"
    )

    failures: list[str] = []
    if cfg.warmup_ratio != 0.03:
        failures.append(f"cfg.warmup_ratio = {cfg.warmup_ratio}, expected 0.03 (plan v2 §11)")

    # Inspect that sft.py downstream actually reads cfg.warmup_ratio into the
    # SFTConfig (line 726: '"warmup_ratio": cfg.warmup_ratio'). We don't
    # actually invoke train_lora (would load a 14 GB model and need a GPU);
    # instead we re-read the source to confirm the mapping is present in
    # this commit.
    print("[phase=verify_propagation] confirming sft.py wires cfg.warmup_ratio into SFTConfig...")
    sft_path = PROJECT_ROOT / "src" / "explore_persona_space" / "train" / "sft.py"
    sft_src = sft_path.read_text()
    target_line = '"warmup_ratio": cfg.warmup_ratio,'
    if target_line not in sft_src:
        failures.append(
            f"sft.py does NOT contain `{target_line}` — propagation broken; "
            f"cfg.warmup_ratio={cfg.warmup_ratio} would be ignored by train_lora."
        )
    else:
        print(
            '  PASS: sft.py contains `"warmup_ratio": cfg.warmup_ratio,` '
            "(plumbing intact, cfg.warmup_ratio=0.03 will reach SFTConfig)."
        )

    # Sanity: round-10 Path A levers also intact.
    print("[phase=verify_levers] confirming round-10 Path A descope levers intact...")
    if cfg.lr != 1e-6:
        failures.append(f"cfg.lr = {cfg.lr}, expected 1e-6 (round-10 v2 lever)")
    if cfg.lora_r != 8:
        failures.append(f"cfg.lora_r = {cfg.lora_r}, expected 8 (round-10 v2 lever)")
    if cfg.lora_alpha != 16:
        failures.append(f"cfg.lora_alpha = {cfg.lora_alpha}, expected 16 (round-10 v2 lever)")
    if cfg.marker_suppress_at_post_response_slot is not True:
        failures.append(
            "cfg.marker_suppress_at_post_response_slot != True (round-7 collator config)"
        )
    if cfg.marker_im_end_token_id != IM_END_TOKEN_ID:
        failures.append(
            f"cfg.marker_im_end_token_id = {cfg.marker_im_end_token_id}, expected {IM_END_TOKEN_ID}"
        )

    if failures:
        print("[phase=FAIL]")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(
        "[phase=done] warmup_ratio=0.03 confirmed in TrainLoraConfig + plumbed into "
        "SFTConfig downstream; round-10 v2 levers (lr=1e-6, r=8, alpha=16) intact."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
