#!/usr/bin/env python3
"""Issue #545 — predictor extraction runner (P3, 1 GPU).

Thin subprocess wrapper around
``behavior_testbed_545.predictors.extract_all`` so the dispatcher's GPU work
stays subprocess-isolated (one framework per process).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

if Path("/workspace").exists():  # pod-only cache redirect; VM keeps its default
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #545 predictor extraction (Groups A-D)")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--skip-gpu", action="store_true", help="CPU-only groups (D + base prior)")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    from explore_persona_space.experiments.behavior_testbed_545.predictors import extract_all

    out = extract_all(skip_gpu=args.skip_gpu)
    print(f"[phase=done] predictors at {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
