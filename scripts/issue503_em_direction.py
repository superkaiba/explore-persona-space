#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (×, →, —, ρ) in scientific docstrings + logs.
"""Issue #503 — Bucket D H7-7c convergent-EM-direction projection CLI
(plan v2 §4.5).

Per plan §4.5 mechanism projection — DESCRIPTIVE per MF-8(a) + MF-7:

For each fine-tuned adapter in {D0, D1, D2, D3, D4} × 3 seeds = 15
adapter-level data points, this script:

1. Loads precomputed residual shifts (post-FT minus pre-FT) at the
   broad-EM in-context prompt position, L25 / p5.
2. Loads the Soligo et al. rank-1 LoRA EM direction extracted from a
   matched insecure-code-trained adapter (validated to ablate the EM
   per plan §12 #22 — implementation detail external to this script).
3. Loads the Soligo non-EM persona directions (educational +
   secure_code) for MF-7(b).
4. Computes per-(selector, seed) H7-7c verdict per
   ``em_direction.h7_7c_verdict``: cosine_em, random CI from MF-7(a),
   non-EM max from MF-7(b), descriptive mechanism-share read.

The output is purely descriptive — no threshold gate, no p-value, NOT
in the H8 calibration headline. Per MF-8(a), the analyzer reads it as
"qualitative signal" alongside the headline H7-7b rank correlation.

Required input artifacts (passed via --shift-dir + --direction-dir):

  shift_dir/{selector}_seed{seed}.npy   # (d_model,) per-cell shift
  direction_dir/em_convergent.npy        # (d_model,) EM rank-1 direction
  direction_dir/non_em_educational.npy   # (d_model,) educational
  direction_dir/non_em_secure_code.npy   # (d_model,) secure_code

Outputs:

  eval_results/issue503/em_direction/h7_7c_verdicts.jsonl
    -- one row per (selector, seed) with all projection bars + verdict
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("issue503.em_direction")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def output_dir(repo_root: Path) -> Path:
    p = repo_root / "eval_results" / "issue503" / "em_direction"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_direction(path: Path, kind: str, *, layer: int, position_name: str):
    from explore_persona_space.experiments.issue503.em_direction import RankOneDirection

    if not path.exists():
        raise FileNotFoundError(f"Direction file missing: {path}")
    vec = np.load(path)
    if vec.ndim != 1:
        raise ValueError(f"Direction {path} must be 1-D; got {vec.shape}")
    # Normalize to unit vector (Soligo extractor already does this; harmless).
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return RankOneDirection(
        kind=kind,
        layer=layer,
        position_name=position_name,
        direction=vec,
        alpha=1.0,
        source_label=str(path.stem),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shift-dir",
        type=Path,
        required=True,
        help="Dir with per-(selector, seed) shift .npy files.",
    )
    parser.add_argument(
        "--direction-dir",
        type=Path,
        required=True,
        help="Dir with em_convergent.npy + non_em_*.npy direction files.",
    )
    parser.add_argument("--layer", type=int, default=25)
    parser.add_argument("--position-name", default="p5")
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=["D0_random", "D1_representation", "D2_gradient", "D3_cosine", "D4_format"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 42, 137])
    parser.add_argument(
        "--n-random-directions",
        type=int,
        default=16,
        help="MF-7(a): number of norm-matched random baselines.",
    )
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args(argv)

    from explore_persona_space.experiments.issue503.em_direction import (
        ResidualShift,
        h7_7c_disclaimer,
        h7_7c_verdict,
    )

    root = repo_root()
    shift_dir = args.shift_dir.resolve()
    direction_dir = args.direction_dir.resolve()

    em_direction = load_direction(
        direction_dir / "em_convergent.npy",
        kind="em_convergent",
        layer=args.layer,
        position_name=args.position_name,
    )
    non_em_dirs = []
    for kind_label, fname in (
        ("non_em_educational", "non_em_educational.npy"),
        ("non_em_secure_code", "non_em_secure_code.npy"),
    ):
        non_em_path = direction_dir / fname
        if non_em_path.exists():
            non_em_dirs.append(
                load_direction(
                    non_em_path,
                    kind=kind_label,
                    layer=args.layer,
                    position_name=args.position_name,
                )
            )
        else:
            logger.warning("Non-EM direction %s missing; skipping that baseline.", non_em_path)

    out_dir = output_dir(root)
    out_path = out_dir / "h7_7c_verdicts.jsonl"

    n_written = 0
    with out_path.open("w") as fout:
        for selector_id in args.selectors:
            for seed in args.seeds:
                shift_path = shift_dir / f"{selector_id}_seed{seed}.npy"
                if not shift_path.exists():
                    logger.warning("Shift file missing: %s — skipping", shift_path)
                    continue
                delta = np.load(shift_path)
                if delta.ndim != 1 or delta.shape[0] != em_direction.direction.shape[0]:
                    raise ValueError(
                        f"Shift {shift_path} shape {delta.shape}; "
                        f"expected ({em_direction.direction.shape[0]},)"
                    )
                shift = ResidualShift(
                    selector_id=selector_id,
                    seed=seed,
                    layer=args.layer,
                    position_name=args.position_name,
                    delta=delta,
                    n_probes=1,  # caller has averaged already
                )
                verdict = h7_7c_verdict(
                    shift,
                    em_direction,
                    non_em_dirs,
                    n_random_directions=args.n_random_directions,
                    rng_seed=args.rng_seed,
                )
                row = asdict(verdict)
                # numpy types → Python types for clean JSON
                row["cosine_em"] = float(row["cosine_em"])
                row["cosine_random_ci_upper"] = float(row["cosine_random_ci_upper"])
                row["cosine_non_em_max"] = float(row["cosine_non_em_max"])
                row["per_direction"] = {k: float(v) for k, v in row["per_direction"].items()}
                fout.write(json.dumps(row) + "\n")
                n_written += 1

    # Disclaimer file alongside output for the analyzer.
    disclaimer_path = out_dir / "h7_7c_disclaimer.txt"
    disclaimer_path.write_text(h7_7c_disclaimer() + "\n")

    print(f"Wrote {n_written} H7-7c verdicts to {out_path}")
    print(f"Disclaimer (MF-8(a)): {h7_7c_disclaimer()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
