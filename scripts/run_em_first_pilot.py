#!/usr/bin/env python3
"""EM-first flipped protocol pilot (#84 follow-up).

Flips the marker-transfer protocol:
  Original #80/#83/#84:  couple source+[ZLT]  →  EM  →  does marker transfer to assistant?
  Flipped pilot:         EM  →  couple evil_ai+[ZLT] on the EM-broken model
                              →  does the coupling bleed to assistant?

This is a direct test of Wang et al.'s "EM collapses personas into one villain"
hypothesis: if EM merges evil_ai and assistant at the representation level,
then coupling trained post-EM on evil_ai should also fire on the assistant
persona — the bleed would be a property of the post-EM geometry itself, not of
EM reading off a pre-existing coupled feature.

Stages
------
1. Merge /workspace/marker_transfer_issue84/c2/em_lora_seed{seed} onto
   Qwen/Qwen2.5-7B-Instruct  →  /workspace/em_first_pilot/em_base_seed{seed}/
   (c2 is "raw base + EM"; its em_lora was trained on pristine Qwen in the
   #84 Wave 1, so this merge reconstructs the EM-broken base model.)

2. Shell out to run_single_token_multi_source.py with --base_model pointing
   at em_base; the script handles coupling data gen, coupling LoRA training
   (marker-only loss on [ZLT], lr=5e-6, 20ep, r=32), merge, and marker eval
   on 12 personas x 28 questions x 10 completions.

3. Parse the resulting run_result.json and print a comparison vs the #84 C4
   baseline (pristine base + evil_ai coupling, no EM).

Usage
-----
    # Single seed pilot:
    nohup uv run python scripts/run_em_first_pilot.py --seed 42 --gpu 0 \
        > /workspace/logs/em_first_pilot/seed42.log 2>&1 &

    # Reuse existing em_base if already merged:
    uv run python scripts/run_em_first_pilot.py --seed 42 --gpu 0 --skip_merge
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORK_ROOT = Path("/workspace/em_first_pilot")
MARKER_TRANSFER_ROOT = Path("/workspace/marker_transfer_issue84")
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

log = logging.getLogger("em_first_pilot")


def setup_logging() -> None:
    log.setLevel(logging.INFO)
    if log.handlers:
        return
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    log.addHandler(h)


def stage_merge_em(seed: int, gpu: int, out_dir: Path) -> None:
    """Merge c2's EM LoRA from #84 Wave 1 onto pristine Qwen."""
    adapter = MARKER_TRANSFER_ROOT / "c2" / f"em_lora_seed{seed}"
    if not (adapter / "adapter_model.safetensors").exists():
        raise FileNotFoundError(
            f"Missing EM LoRA at {adapter}. Run #84 Wave 1 c2 seed={seed} first."
        )
    if (out_dir / "config.json").exists():
        log.info("em_base already exists at %s — skipping merge", out_dir)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Merging EM LoRA %s onto %s → %s", adapter, BASE_MODEL_ID, out_dir)
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from explore_persona_space.train.sft import merge_lora

    merge_lora(BASE_MODEL_ID, str(adapter), str(out_dir), gpu_id=gpu)
    log.info("Merge done in %.1fs", 0.0)


def stage_train_and_eval(seed: int, gpu: int, base_model_dir: Path, output_dir: Path) -> Path:
    """Invoke run_single_token_multi_source.py to train coupling + eval on em_base."""
    cmd = [
        "uv",
        "run",
        "python",
        str(PROJECT_ROOT / "scripts" / "run_single_token_multi_source.py"),
        "--source",
        "evil_ai",
        "--seed",
        str(seed),
        "--gpu",
        str(gpu),
        "--base_model",
        str(base_model_dir),
        "--output_dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    env["WANDB_PROJECT"] = "em_first_pilot"
    log.info("Launching coupling trainer: %s", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"Coupling stage failed with code {r.returncode}")
    return output_dir / "run_result.json"


def report(result_path: Path) -> None:
    """Print a comparison vs #84 C4 baseline (no-EM evil_ai coupling)."""
    if not result_path.exists():
        log.error("Missing run_result.json at %s", result_path)
        return
    with open(result_path) as f:
        result = json.load(f)

    c4_path = MARKER_TRANSFER_ROOT / "c4" / "run_result_c4_seed42.json"
    if c4_path.exists():
        with open(c4_path) as f:
            c4 = json.load(f)
    else:
        c4 = None

    def rate(per_persona: dict, name: str) -> float:
        v = per_persona.get(name)
        if v is None:
            return float("nan")
        if isinstance(v, dict):
            return v.get("strict_rate", v.get("rate", float("nan")))
        return v

    pilot_pp = result.get("marker_eval", {}).get("per_persona") or result.get("per_persona") or {}
    c4_pp = (c4 or {}).get("marker_eval", {}).get("per_persona", {})

    log.info("=" * 72)
    log.info("EM-first pilot vs #84 C4 (no-EM baseline)  — [ZLT] strict rate")
    log.info("=" * 72)
    log.info(f"{'Persona':<25} {'Pilot (em→couple)':>18} {'C4 (couple only)':>18}")
    for persona in sorted(set(pilot_pp) | set(c4_pp)):
        p = rate(pilot_pp, persona)
        c = rate(c4_pp, persona)
        marker = "  <-- TARGET" if persona in ("assistant", "helpful_assistant") else ""
        log.info(f"{persona:<25} {100 * p:>17.2f}% {100 * c:>17.2f}%{marker}")
    log.info("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--skip_merge",
        action="store_true",
        help="Reuse existing em_base_seed{seed}/ if already merged.",
    )
    args = parser.parse_args()

    setup_logging()

    t0 = time.time()
    em_base = WORK_ROOT / f"em_base_seed{args.seed}"
    coupling_out = WORK_ROOT / f"coupling_evil_ai_seed{args.seed}"

    if not args.skip_merge:
        stage_merge_em(args.seed, args.gpu, em_base)
    elif not (em_base / "config.json").exists():
        raise FileNotFoundError(f"--skip_merge but {em_base} does not exist")

    result_path = stage_train_and_eval(args.seed, args.gpu, em_base, coupling_out)
    report(result_path)
    log.info("PILOT DONE in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
