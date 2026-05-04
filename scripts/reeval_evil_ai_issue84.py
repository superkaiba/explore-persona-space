#!/usr/bin/env python3
"""Re-run marker eval for the existing evil_ai adapter (issue #84).

The first run's eval was executed on stale code that didn't inject
`evil_ai` into the persona set, so source-rate was never measured.
This script:
  1. Re-merges the adapter (already on disk) with the base model.
  2. Runs the marker + capability eval INCLUDING evil_ai in the persona
     set (via the commit-7c22ba6 fix).
  3. Overwrites run_result.json with the correct source_marker.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_leakage_v3_onpolicy import ALL_EVAL_PERSONAS, run_eval  # noqa: E402
from run_single_token_multi_source import BASE_MODEL  # noqa: E402

from explore_persona_space.personas import EVIL_AI_PROMPT  # noqa: E402
from explore_persona_space.train.sft import merge_lora  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="/workspace/marker_transfer_issue84/evil_ai_seed42",
        help="Existing experiment dir with adapter/",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    exp_dir = Path(args.exp_dir)
    adapter_dir = exp_dir / "adapter"
    merged_dir = exp_dir / "merged"
    result_path = exp_dir / "run_result.json"

    if not adapter_dir.exists():
        raise SystemExit(f"Adapter not found: {adapter_dir}")

    print(f"[reeval] merging {adapter_dir} -> {merged_dir}")
    t0 = time.time()
    merge_lora(
        base_model_path=BASE_MODEL,
        adapter_path=str(adapter_dir),
        output_dir=str(merged_dir),
        gpu_id=args.gpu,
    )
    print(f"[reeval] merged in {(time.time() - t0) / 60:.1f} min")

    eval_personas = dict(ALL_EVAL_PERSONAS)
    eval_personas["evil_ai"] = EVIL_AI_PROMPT
    print(f"[reeval] persona set ({len(eval_personas)}): {sorted(eval_personas.keys())}")

    eval_results = run_eval(
        merged_path=str(merged_dir),
        output_dir=exp_dir,
        gpu_id=args.gpu,
        personas=eval_personas,
    )

    marker_rates = eval_results.get("marker", {})
    src_rate = marker_rates.get("evil_ai", 0.0)
    bystander = {k: v for k, v in marker_rates.items() if k != "evil_ai"}
    max_bystander = max(bystander.values()) if bystander else 0.0

    prev = {}
    if result_path.exists():
        with open(result_path) as f:
            prev = json.load(f)

    result = {
        "config": {
            "lr": args.lr,
            "epochs": args.epochs,
            "source": "evil_ai",
            "seed": args.seed,
            "marker_tail_tokens": 0,
        },
        "loss": prev.get("loss"),
        "eval": eval_results,
        "wall_minutes": prev.get("wall_minutes"),
        "source_marker": src_rate,
        "max_bystander_marker": max_bystander,
        "per_persona_marker": marker_rates,
        "reeval_note": (
            "Re-eval from existing adapter with evil_ai persona injected. "
            "Original eval missed evil_ai (stale code)."
        ),
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[reeval] saved: {result_path}")

    print("\n" + "=" * 80)
    print(f"EVIL_AI RE-EVAL — source={src_rate:.1%}  max_bystander={max_bystander:.1%}")
    print("=" * 80)
    for p, r in sorted(marker_rates.items()):
        marker = " *" if p == "evil_ai" else ""
        print(f"  {p:<25} {r:.1%}{marker}")

    if merged_dir.exists():
        shutil.rmtree(merged_dir)
        print(f"[reeval] cleaned {merged_dir}")


if __name__ == "__main__":
    main()
