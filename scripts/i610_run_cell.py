#!/usr/bin/env python
"""Task #610 — per-(cell, seed) subprocess entrypoint (spawned by i610_dispatch.py).

Identical body to the #600 runner via the extended ``run_one_cell`` with the
 #610 kwargs: the spec is REBUILT in-process from the PARENT manifest via
``build_610_spec`` (``spec_override`` bypasses the #600 registry, whose
qwen_default-exactly-once assert is structurally incompatible with the
no-default arm), ``extra_eval_personas`` adds the primary DV (qwen_default)
plus the cluster-identity probe (assistant) to the eval set, and the HF /
WandB identifiers carry the chassis prefixes.

Chassis (v2 plan §4.3): the config is RESOLVED FROM ``--cell`` (each chassis
has a unique new-cell slug), so the dispatcher's subprocess command shape is
unchanged. An explicit ``--chassis`` is accepted and must agree with the
slug-resolved config (fail-loud on mismatch).

GPU pinning contract: the dispatcher exports CUDA_VISIBLE_DEVICES=<gpu> in
THIS process's environment AND passes --gpu-id <gpu>, so sft.py's in-process
clobber rewrites the same value (gotcha #545).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# uv run python does NOT auto-load .env (HF_TOKEN is needed for the inline
# adapter upload + base-model load).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from explore_persona_space.experiments.default_dose_610 import (  # noqa: E402
    CHASSES,
    EPOCHS_PINNED,
    EXTRA_EVAL_PERSONAS,
    WANDB_PROJECT,
    chassis_for_slug,
)
from explore_persona_space.experiments.default_dose_610.cells import (  # noqa: E402
    build_610_spec,
)
from explore_persona_space.experiments.targeted_proximity_600.cells import (  # noqa: E402
    load_manifest,
)
from explore_persona_space.experiments.targeted_proximity_600.dispatch import (  # noqa: E402
    run_one_cell,
)


def main(argv: list[str] | None = None) -> int:
    registered = sorted(c.new_slug for c in CHASSES.values())
    ap = argparse.ArgumentParser(description="Task #610 per-cell runner")
    ap.add_argument("--cell", required=True, help=f"One of the registered slugs: {registered}.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument(
        "--gpu-id",
        type=int,
        required=True,
        help="ASSIGNED PHYSICAL GPU index (must match the launcher's CUDA_VISIBLE_DEVICES).",
    )
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--manifest", type=Path, required=True, help="The PARENT #600 manifest.")
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument(
        "--chassis",
        choices=sorted(CHASSES),
        default=None,
        help="Optional explicit chassis; must agree with the --cell slug resolution.",
    )
    args = ap.parse_args(argv)
    try:
        chassis = chassis_for_slug(args.cell)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc
    if args.chassis is not None and args.chassis != chassis.name:
        raise SystemExit(
            f"--chassis {args.chassis!r} contradicts the --cell slug {args.cell!r} "
            f"(which belongs to chassis {chassis.name!r})."
        )
    if args.epochs != EPOCHS_PINNED:
        # Kill-criterion hardening (plan §7.1): NO epochs ladder — re-pinning
        # would unmatch the reused parent arm's 63 steps and void the contrast.
        raise SystemExit(
            f"--epochs must be the PINNED {EPOCHS_PINNED} (plan §7.1: no epochs ladder; "
            f"matched steps with the reused parent arm are load-bearing); got {args.epochs}"
        )
    # Preset BEFORE run_one_cell (its setdefault never overrides a preset).
    # ONE WandB project across chassis (v2 §6); the run-name prefix carries it.
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    spec = build_610_spec(load_manifest(args.manifest), chassis)
    result = run_one_cell(
        cell_slug=args.cell,
        seed=args.seed,
        gpu_id=args.gpu_id,
        epochs=args.epochs,
        manifest_path=args.manifest,
        output_root=args.output_root or chassis.output_root_default,
        data_root=args.data_root or chassis.data_root_default,
        spec_override=spec,
        extra_eval_personas=EXTRA_EVAL_PERSONAS,
        hf_adapter_prefix=chassis.hf_adapter_path_prefix,
        run_name_prefix=chassis.run_name_prefix,
    )
    print(f"cell complete: {result['cell_slug']}_seed{result['seed']} (eval + persist OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
