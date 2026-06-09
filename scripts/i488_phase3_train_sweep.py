# ruff: noqa: RUF001, RUF002
"""Issue #488 Phase 3 v6 — production train sweep at picked rung recipe.

Plan v6 §4.8 + §9 + §14: after Phase 2 ladder PASSes (PASS or
PICK_AT_SATURATION verdict), this dispatcher:

  1. Reads ``/workspace/logs/issue-488-smoke-result.json`` (the ladder's
     sentinel) to extract the picked rung + recipe.
  2. Fans out the 27 trained LoRA conditions × 2 seeds = 54 cells across
     the pod's 8 GPUs (CVD-shard, +gpu_id Hydra override per CLAUDE.md
     ``feedback_cvd_hydra_override``).
  3. Each cell calls ``i488_phase23_train.py`` with the picked rung's
     (lr, lora_r, lora_alpha, max_rows_per_side, epochs, n_dupes,
     warmup_ratio) threaded through, saving adapters at ALL 6 production
     fracs ∈ {0.10, 0.25, 0.50, 1.00, 2.00, 3.00}.

Per CLAUDE.md "Checkpoint per phase": each cell's adapter is uploaded
to HF (via ``i488_phase23_train.py``'s ``FractionAdapterSaveCallback``)
the moment its frac=3.00 checkpoint completes; failures of later cells
do not orphan earlier ones.

Per CLAUDE.md pod-side rule + ``poll_pipeline.py``: emits ``[phase=...]``
log lines (incl. ``[phase=done]`` on graceful completion) and a final
sentinel at ``/workspace/logs/issue-488-phase3-results.json`` so the
VM-side orchestrator can detect success.

CLI:
    # Standard: read picked recipe from ladder sentinel, sweep all 27×2.
    uv run python scripts/i488_phase3_train_sweep.py

    # Override sentinel + run a subset of cells (for canary / smoke):
    uv run python scripts/i488_phase3_train_sweep.py \\
        --sentinel /tmp/test-sentinel.json \\
        --only-cids A1 G2 --only-seeds 42

    # Resume after partial failure (skip cells whose adapter already on HF):
    uv run python scripts/i488_phase3_train_sweep.py --resume
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

logger = logging.getLogger("i488.phase3")

# 27 trained conditions per plan v3 §5 (unchanged in v6).
ALL_CIDS = [
    "A1", "A2", "A3", "A4", "A5",
    "B1", "B2", "B3", "B4", "B5",
    "C1",
    "D1", "D2", "D3", "D4", "D5",
    "E2", "E3", "E4", "E5",
    "F1", "F2", "F3", "F4",
    "G1", "G2", "G3",
]  # fmt: skip
DEFAULT_SEEDS = [42, 137]
DEFAULT_FRACS = [0.10, 0.25, 0.50, 1.00, 2.00, 3.00]

DEFAULT_SENTINEL_PATH = Path("/workspace/logs/issue-488-smoke-result.json")
RESULTS_SENTINEL_PATH = Path("/workspace/logs/issue-488-phase3-results.json")
LOG_DIR = Path("logs/issue_488/phase3")


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _read_picked_recipe(sentinel_path: Path) -> dict:
    """Read the picked rung's recipe from the ladder's smoke-result sentinel.

    Returns a dict with keys {rung, lr, lora_r, lora_alpha, max_rows_per_side,
    epochs, n_dupes, warmup_ratio, verdict}. Raises if the sentinel is
    missing, malformed, or carries a non-PASS verdict.
    """
    if not sentinel_path.exists():
        raise FileNotFoundError(
            f"Smoke-result sentinel missing at {sentinel_path}. "
            "Run scripts/i488_phase2_ladder.py first."
        )
    raw = json.loads(sentinel_path.read_text())
    if raw.get("kind") != "epm:smoke-result":
        raise RuntimeError(
            f"Sentinel at {sentinel_path} has kind={raw.get('kind')!r}, "
            "expected 'epm:smoke-result'."
        )
    # The marker payload lives under `note` (or `payload` synonym per
    # poll_pipeline.py). It's a JSON string per the ladder's writer.
    note = raw.get("note") or raw.get("payload")
    if isinstance(note, str):
        note = json.loads(note)
    verdict = note.get("verdict")
    if verdict not in {"PASS", "PICK_AT_SATURATION"}:
        raise RuntimeError(
            f"Sentinel verdict={verdict} not in (PASS, PICK_AT_SATURATION); "
            "Phase 3 refuses to launch on a non-PASS ladder result."
        )
    recipe = note.get("recipe") or {}
    required = (
        "lr",
        "lora_r",
        "lora_alpha",
        "max_rows_per_side",
        "epochs",
        "n_dupes",
        "warmup_ratio",
    )
    missing = [k for k in required if k not in recipe]
    if missing:
        raise RuntimeError(f"Sentinel recipe missing keys {missing}; got {sorted(recipe)}.")
    return {
        "rung": note.get("picked_rung"),
        "verdict": verdict,
        **recipe,
    }


def _train_cmd(
    cid: str,
    seed: int,
    fracs: list[float],
    recipe: dict,
    gpu_id: int,
) -> list[str]:
    """Build the subprocess argv for one cell at the picked rung's recipe.

    Per CLAUDE.md ``feedback_cvd_hydra_override``: pass ``--gpu-id`` (which the
    trainer reads to set ``CUDA_VISIBLE_DEVICES``); do NOT rely on env CVD.
    """
    return [
        "uv",
        "run",
        "python",
        "scripts/i488_phase23_train.py",
        "--conds",
        cid,
        "--seeds",
        str(seed),
        "--gpu-id",
        str(gpu_id),
        "--fracs",
        *[str(f) for f in fracs],
        "--lr",
        str(recipe["lr"]),
        "--lora-r",
        str(recipe["lora_r"]),
        "--lora-alpha",
        str(recipe["lora_alpha"]),
        "--max-rows-per-side",
        str(recipe["max_rows_per_side"]),
        "--epochs",
        str(recipe["epochs"]),
        "--n-dupes",
        str(recipe["n_dupes"]),
        "--warmup-ratio",
        str(recipe["warmup_ratio"]),
    ]


def _adapter_uploaded(cid: str, seed: int, frac: float) -> bool:
    """Resume check: is the (cid, seed, frac) adapter already on HF Hub?"""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        files = api.list_repo_files(repo_id="superkaiba1/explore-persona-space")
        # Adapter path convention used by FractionAdapterSaveCallback:
        # adapters/i488_<cid>_seed<seed>_frac<NNN>/adapter_model.safetensors.
        frac_tag = f"frac{round(frac * 100):03d}"
        prefix = f"adapters/i488_{cid}_seed{seed}_{frac_tag}/"
        return any(f.startswith(prefix) and f.endswith("adapter_model.safetensors") for f in files)
    except Exception as e:
        logger.warning("HF resume check failed (non-fatal): %s", e)
        return False


def _run_cell(
    cid: str,
    seed: int,
    fracs: list[float],
    recipe: dict,
    gpu_id: int,
    log_dir: Path,
    resume: bool,
) -> tuple[str, int, str]:
    """Train one (cid, seed) cell at the picked rung's recipe. Blocks until done.

    Returns (cid:seed, rc, log_path).
    """
    cell_log = log_dir / f"cell_{cid}_seed{seed}_gpu{gpu_id}.log"
    cell_log.parent.mkdir(parents=True, exist_ok=True)

    if resume and all(_adapter_uploaded(cid, seed, f) for f in fracs):
        # Skip if all requested fracs are already on HF.
        logger.info("[resume] skipping cell=%s seed=%d — all fracs on HF", cid, seed)
        return f"{cid}:{seed}", 0, str(cell_log)

    cmd = _train_cmd(cid, seed, fracs, recipe, gpu_id)
    env = {**os.environ}
    # MooseFS quota safety per `.claude/rules/gotchas.md`.
    env["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] = "1"

    with open(cell_log, "a") as logf:
        logf.write(f"\n==== {_now()} train cell={cid} seed={seed} gpu={gpu_id} ====\n")
        logf.write(f"cmd: {' '.join(shlex.quote(c) for c in cmd)}\n")
        logf.flush()
        rc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        ).returncode
    return f"{cid}:{seed}", rc, str(cell_log)


def _write_results_sentinel(
    *,
    n_cells_total: int,
    n_cells_ok: int,
    n_cells_failed: int,
    failed_cells: list[str],
    recipe: dict,
    started_at: str,
    ended_at: str,
) -> None:
    """Write the Phase 3 results sentinel for poll_pipeline."""
    RESULTS_SENTINEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:phase3-results",
        "version": 1,
        "issue": 488,
        "ts": _now(),
        "by": "i488_phase3_train_sweep",
        "note": json.dumps(
            {
                "n_cells_total": n_cells_total,
                "n_cells_ok": n_cells_ok,
                "n_cells_failed": n_cells_failed,
                "failed_cells": failed_cells,
                "recipe": recipe,
                "started_at": started_at,
                "ended_at": ended_at,
                "plan_version": "v6",
            },
            indent=2,
        ),
    }
    RESULTS_SENTINEL_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("Wrote Phase 3 results sentinel → %s", RESULTS_SENTINEL_PATH)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--sentinel",
        type=Path,
        default=DEFAULT_SENTINEL_PATH,
        help="Path to the ladder's smoke-result sentinel.",
    )
    ap.add_argument(
        "--only-cids",
        nargs="+",
        default=None,
        help="Subset of source cids (default = all 27).",
    )
    ap.add_argument(
        "--only-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Subset of seeds (default = 42 137).",
    )
    ap.add_argument(
        "--only-fracs",
        nargs="+",
        type=float,
        default=None,
        help="Subset of fracs (default = 0.10 0.25 0.50 1.00 2.00 3.00).",
    )
    ap.add_argument(
        "--n-gpus",
        type=int,
        default=8,
        help="Number of CVD-shard parallel cells (one cell per GPU).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip cells whose adapter is already on HF Hub.",
    )
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=LOG_DIR,
    )
    args = ap.parse_args()

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    args.log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read picked recipe from ladder sentinel.
    logger.info("[phase=phase3_load_recipe]")
    try:
        recipe = _read_picked_recipe(args.sentinel)
    except Exception as e:
        logger.error("Failed to load picked recipe: %s", e)
        return 2
    logger.info(
        "Picked rung=%s verdict=%s recipe=lr=%s r=%s α=%s rows=%s epochs=%s n_dupes=%s warmup=%s",
        recipe["rung"],
        recipe["verdict"],
        recipe["lr"],
        recipe["lora_r"],
        recipe["lora_alpha"],
        recipe["max_rows_per_side"],
        recipe["epochs"],
        recipe["n_dupes"],
        recipe["warmup_ratio"],
    )

    # 2. Build work list.
    cids = args.only_cids or ALL_CIDS
    seeds = args.only_seeds or DEFAULT_SEEDS
    fracs = args.only_fracs or DEFAULT_FRACS
    work_list: list[tuple[str, int]] = [(c, s) for c in cids for s in seeds]
    logger.info(
        "[phase=phase3_dispatch] %d cells (cids=%d × seeds=%d) × %d fracs across %d GPUs",
        len(work_list),
        len(cids),
        len(seeds),
        len(fracs),
        args.n_gpus,
    )

    started_at = _now()

    # 3. Round-robin dispatch across GPUs, n_gpus parallel at a time.
    # Use multiprocessing.Pool — simpler than asyncio for blocking subprocess work.
    from multiprocessing import Pool

    cell_args: list[tuple[str, int, list[float], dict, int, Path, bool]] = []
    for i, (cid, seed) in enumerate(work_list):
        gpu_id = i % args.n_gpus
        cell_args.append((cid, seed, fracs, recipe, gpu_id, args.log_dir, args.resume))

    n_ok = 0
    failed_cells: list[str] = []
    with Pool(processes=args.n_gpus) as pool:
        for cell_id, rc, log_path in pool.starmap(_run_cell, cell_args):
            if rc == 0:
                n_ok += 1
                logger.info("[cell_done] %s rc=0 log=%s", cell_id, log_path)
            else:
                failed_cells.append(cell_id)
                logger.error("[cell_FAILED] %s rc=%d log=%s", cell_id, rc, log_path)

    ended_at = _now()
    n_total = len(work_list)
    n_failed = len(failed_cells)

    _write_results_sentinel(
        n_cells_total=n_total,
        n_cells_ok=n_ok,
        n_cells_failed=n_failed,
        failed_cells=failed_cells,
        recipe=recipe,
        started_at=started_at,
        ended_at=ended_at,
    )

    if n_failed > 0:
        logger.error(
            "[phase=phase3_failed] %d/%d cells failed: %s",
            n_failed,
            n_total,
            failed_cells,
        )
        return 2

    logger.info("[phase=phase3_ok] %d/%d cells trained", n_ok, n_total)
    logger.info("[phase=done]")
    # Suppress unused-import lint on the time module (kept for explicit
    # subprocess polling future use; harmless to leave).
    _ = time
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
