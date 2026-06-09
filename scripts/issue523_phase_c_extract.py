#!/usr/bin/env python3
"""Issue #523 — Phase C held-out distance extraction (thin wrapper).

Invokes ``scripts/issue502_dispatch.py`` with #523-specific paths:

  * --probe-pool   eval_results/issue_523/heldout_probes_500.json
  * --bakeoff-root eval_results/issue_523/bakeoff
  * --figures-root figures/issue_523/bakeoff
  * --class-d-extension-path eval_results/issue_523/class_d_rewrites_extended_v1.json

The extraction runs ONCE against the base Qwen-2.5-7B-Instruct (no LoRA loaded;
the bake-off extractor never loads adapters per plan v2 §4 Phase C). Both the
seed-42 and seed-43 legs of Phase D reuse this single set of distance matrices
— only the regression target (ΔG) swaps between seeds.

Usage::

    # Full extraction (4× H100 pod, ~2.25 h).
    uv run python scripts/issue523_phase_c_extract.py

    # Smoke (1 cond A1, 12 probes, 1 layer).
    uv run python scripts/issue523_phase_c_extract.py \\
        --smoke-cell A1 --probes 12

    # Custom layers / num-gpus.
    uv run python scripts/issue523_phase_c_extract.py --num-gpus 4 --batch-size 8
"""

# Greek + special characters appear in docstrings / comments.
# ruff: noqa: RUF002

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

logger = logging.getLogger("i523.phase_c")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROBE_POOL = PROJECT_ROOT / "eval_results" / "issue_523" / "heldout_probes_500.json"
SMOKE_PROBE_POOL = PROJECT_ROOT / "eval_results" / "issue_523" / "heldout_probes_500.smoke.json"
CLASS_D_EXT = PROJECT_ROOT / "eval_results" / "issue_523" / "class_d_rewrites_extended_v1.json"
SMOKE_CLASS_D_EXT = (
    PROJECT_ROOT / "eval_results" / "issue_523" / "class_d_rewrites_extended_v1.smoke.json"
)
BAKEOFF_ROOT = PROJECT_ROOT / "eval_results" / "issue_523" / "bakeoff"
SMOKE_BAKEOFF_ROOT = PROJECT_ROOT / "eval_results" / "issue_523" / "bakeoff" / "smoke"
FIGURES_ROOT = PROJECT_ROOT / "figures" / "issue_523" / "bakeoff"


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase C extraction wrapper for the #523 held-out probe pool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="GPU count; 0 = auto-detect via nvidia-smi.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Probes per generate() batch.",
    )
    p.add_argument(
        "--probes",
        type=int,
        default=500,
        help="How many probes to use from the held-out pool (default 500).",
    )
    p.add_argument(
        "--smoke-cell",
        default=None,
        help=(
            "When set, run a one-cond smoke (e.g. A1). Combined with --probes <small> "
            "this exercises the full code path end-to-end with minimal compute."
        ),
    )
    p.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=list(range(28)),
        help="Residual-stream layers to extract / score (default 0..27).",
    )
    p.add_argument(
        "--smoke-layers",
        nargs="+",
        type=int,
        default=[0, 22],
        help="Layers to extract in smoke mode (default 0 and 22 — fast).",
    )
    p.add_argument(
        "--extra-args",
        nargs="*",
        default=[],
        help="Additional positional args passed through to issue502_dispatch.py.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    smoke = args.smoke_cell is not None
    probe_pool = SMOKE_PROBE_POOL if smoke else PROBE_POOL
    class_d_ext = SMOKE_CLASS_D_EXT if smoke else CLASS_D_EXT
    bakeoff_root = SMOKE_BAKEOFF_ROOT if smoke else BAKEOFF_ROOT
    figures_root = FIGURES_ROOT / "smoke" if smoke else FIGURES_ROOT
    layers = args.smoke_layers if smoke else args.layers

    # ── Preflight: assert no LoRA env contamination ──
    # Per plan v2 §4 Phase C (the activation extraction is on the BASE model;
    # ΔG itself is the seed-dependent quantity, not the activations).
    if os.environ.get("EPM_LOAD_LORA"):
        raise RuntimeError(
            "EPM_LOAD_LORA is set in the environment but Phase C extraction "
            "must be on the BASE model only (plan v2 §4 Phase C). Unset it "
            "before re-launching."
        )

    # ── Fail-loud preflight on probe pool ──
    if not probe_pool.exists():
        raise FileNotFoundError(
            f"Probe pool {probe_pool} not found; run Phase A first "
            f"(scripts/issue523_phase_a_generate_probes.py {'--smoke-only' if smoke else ''})."
        )
    if not class_d_ext.exists():
        raise FileNotFoundError(
            f"Class-D extension {class_d_ext} not found; Phase A must produce it."
        )

    bakeoff_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv",
        "run",
        "python",
        str(PROJECT_ROOT / "scripts" / "issue502_dispatch.py"),
        "--probe-pool",
        str(probe_pool),
        "--bakeoff-root",
        str(bakeoff_root),
        "--figures-root",
        str(figures_root),
        "--class-d-extension-path",
        str(class_d_ext),
        "--batch-size",
        str(args.batch_size),
        "--n-probes",
        str(args.probes),
        "--layers",
        *[str(L) for L in layers],
    ]
    if args.num_gpus > 0:
        cmd += ["--num-gpus", str(args.num_gpus)]
    if smoke:
        # Smoke: just the one transformation, force --num-gpus to 1 unless the
        # caller already overrode it (we don't want to spread 1 transformation
        # across N GPUs).
        cmd += ["--transformations", args.smoke_cell]
        if args.num_gpus <= 0:
            cmd += ["--num-gpus", "1"]

    cmd += list(args.extra_args)

    # #523's held-out pool deliberately omits the 50-question q_test prefix
    # that #502's pool carries (Phase A asserts disjointness from q_train +
    # q_test + #502 pool). Tell the bake-off extractor to skip Constraint 1
    # (q_test-prefix bit-identical) and the cosine strict-prefix
    # cross-check against #406. Default-off env switch ⇒ legacy #502
    # reruns are byte-for-byte unaffected.
    os.environ["EPM_PROBE_POOL_STANDALONE"] = "1"

    logger.info("Phase C: invoking %s", " ".join(cmd))
    rc = subprocess.call(
        cmd,
        cwd=PROJECT_ROOT,
        env={**os.environ},  # epm-lint: subprocess explicit env
    )
    if rc != 0:
        logger.error("Phase C dispatcher exited rc=%d", rc)
        return rc
    logger.info(
        "Phase C complete: distance matrices under %s, regression JSONs under %s",
        bakeoff_root / "metrics",
        bakeoff_root / "regression",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
