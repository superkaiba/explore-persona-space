"""Issue #2221 P4 — 24 rs-LoRA fine-tunes on the REAL-data mixes (reused #778 driver).

Thin dispatcher around ``issue778_finetune.run_wave_dispatch`` (the recipe —
r=32/alpha=64 rsLoRA, lr=1e-5, 1 epoch, batch 2 x accum 8, all-7 target
modules, response-only loss, seed 0 — is reused VERBATIM). The ONE declared
override: adapter-only intermediate checkpoints at {10, 25, 50}% of steps
(``--save-fracs``, the opt-in #2221 callback in the reused driver).

Fan-out: one adapter per GPU, ``CUDA_VISIBLE_DEVICES`` pinned in the LAUNCHER
env per cell (the +gpu_id/CVD pattern — the in-process clobber is defeated by
import-time cuInit, gotchas.md), waves of the visible GPU count (4x H100 -> 6
waves of 4).

After training, adapters + frac-checkpoints upload to the HF model repo under
``issue2221_realtwin/adapters/{family}_{version}/`` (``--remine`` ->
``adapters_remine/`` — a constant-composed prefix flip, never a free-form
prefix arg (the #1005 clobber shape), so the parent's adapters are never
overwritten; mirrors ``issue2221_build_mix.py``'s ``train_remine/`` routing).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue778_finetune as ft  # noqa: E402
import issue778_lib as lib  # noqa: E402

from explore_persona_space.experiments.issue_2221 import constants as C  # noqa: E402

logger = logging.getLogger("issue2221.sweep")


def cells_for(args) -> list[tuple[str, str]]:
    """The (family, version) cell list (full 24, or an explicit subset)."""
    if args.cells:
        out = []
        for spec in args.cells:
            family, version = spec.split("/", 1)
            assert family in C.FAMILIES, family
            assert version in C.VERSIONS, version
            out.append((family, version))
        return out
    return [(f, v) for f in C.FAMILIES for v in C.VERSIONS]


def cell_complete(
    ckpt_root: Path, family: str, version: str, fracs: tuple[float, ...] | None
) -> bool:
    """True when the cell's FINAL adapter and EVERY expected frac checkpoint exist.

    The P4 resume predicate (review blocker 4): a crash at cell k must not
    retrain the completed k-1 cells (~0.5 GPU-h each).
    """
    root = ckpt_root / f"{family}_{version}"
    if not (root / "adapter_config.json").is_file():
        return False
    for frac in fracs or ():
        ck = root / f"checkpoint_frac{int(round(frac * 100))}"
        if not (ck / "adapter_config.json").is_file():
            return False
    return True


def pending_cells(
    ckpt_root: Path,
    cells: list[tuple[str, str]],
    fracs: tuple[float, ...] | None,
    *,
    force: bool = False,
) -> tuple[list[tuple[str, str]], list[str]]:
    """(cells to train, skipped-complete cell slugs) — ``force`` retrains all."""
    if force:
        return list(cells), []
    pending, skipped = [], []
    for family, version in cells:
        if cell_complete(ckpt_root, family, version, fracs):
            skipped.append(f"{family}_{version}")
        else:
            pending.append((family, version))
    return pending, skipped


def upload_adapters(args, cells: list[tuple[str, str]]) -> None:
    """Per-cell adapter (+ frac-checkpoint) upload to the HF model repo."""
    from explore_persona_space.orchestrate import hub

    adapters_prefix = f"{C.HF_PREFIX}/{'adapters_remine' if args.remine else 'adapters'}"
    ckpt_root = Path(args.ckpt_root)
    for family, version in cells:
        cell = f"{family}_{version}"
        local = ckpt_root / cell
        if not (local / "adapter_config.json").is_file():
            raise FileNotFoundError(f"adapter missing for {cell}: {local}")
        url = hub._upload(
            local,
            C.HF_MODEL_REPO,
            "model",
            f"{adapters_prefix}/{cell}",
            raise_on_error=True,
        )
        lib.log_phase("p4_upload", f"{cell} -> {url}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset-root", default="data/issue_2221/dataset")
    ap.add_argument("--ckpt-root", default="checkpoints/issue_2221")
    ap.add_argument("--cells", nargs="*", default=None, help="subset 'family/version' specs")
    ap.add_argument(
        "--n-gpus", type=int, default=None, help="wave-size ceiling (default: detected)"
    )
    ap.add_argument("--max-steps", type=int, default=None, help="cap training steps (smoke)")
    ap.add_argument(
        "--save-fracs",
        default=",".join(str(f) for f in C.CHECKPOINT_FRACS),
        help="adapter-only checkpoint fractions (default: the plan's 0.1,0.25,0.5)",
    )
    ap.add_argument("--model", default=lib.MODEL_NAME)
    ap.add_argument("--cpu-only", action="store_true", help="deliberate CPU smoke")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--remine",
        action="store_true",
        help="upload to adapters_remine/ instead of adapters/ (specialized_corpus_remine round)",
    )
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--force", action="store_true", help="retrain cells whose adapters exist")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        from explore_persona_space.orchestrate import hub  # noqa: F401

        # Deferred-import execution for the reused driver's training path.
        from peft import LoraConfig  # noqa: F401
        from transformers import TrainerCallback  # noqa: F401
        from trl import SFTConfig, SFTTrainer  # noqa: F401

        print("[import-check] OK")
        raise SystemExit(0)

    cells = cells_for(args)
    fracs = ft.parse_save_fracs(args.save_fracs)
    train_cells, skipped = pending_cells(Path(args.ckpt_root), cells, fracs, force=args.force)
    if skipped:
        lib.log_phase(
            "p4_finetune",
            f"resume: {len(skipped)} complete cell(s) skipped (adapter + frac "
            f"checkpoints present; --force retrains): {skipped}",
        )
    if train_cells:
        if args.dry_run:
            wave_size = max(args.n_gpus, 1) if args.n_gpus else 8
        else:
            wave_size = ft._compute_wave_size(args.cpu_only, args.n_gpus)
        res = ft.run_wave_dispatch(
            train_cells,
            Path(args.dataset_root),
            Path(args.ckpt_root),
            wave_size=wave_size,
            max_steps=args.max_steps,
            dry_run=args.dry_run,
            model_name=args.model,
            cpu_only=args.cpu_only,
            save_fracs=args.save_fracs,
        )
    else:
        res = {"cells": [], "note": "all requested cells complete — nothing to train"}
    print(json.dumps({"phase": "p4_finetune", "skipped_complete": skipped, **res}, indent=2))
    if not (args.dry_run or args.no_upload):
        upload_adapters(args, cells)  # full requested set — idempotent re-upload covers resumes
        lib.write_results_sentinel(
            C.ISSUE,
            "p4_finetune",
            1,
            f"{len(train_cells)} cells trained (+{len(skipped)} resumed) + adapters uploaded",
        )
    lib.log_phase("p4", "done")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
