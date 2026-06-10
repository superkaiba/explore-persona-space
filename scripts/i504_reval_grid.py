# ruff: noqa: RUF003  # em-dash + Qwen marker " ※" + Greek ΔG + minus sign − intentional
#!/usr/bin/env python3
"""Task #504 recovery re-eval driver — re-evaluate BOTH already-trained #504 cells.

Re-evaluates the two adapters on disk under
``/workspace/runs/issue_504/c504_smoke_r{4,8}_seed42/`` (mirror on HF Hub:
``superkaiba1/explore-persona-space/adapters/issue_504/c504_smoke_r{4,8}_seed42``)
using the production #472 eval rig (``run_trajectory_eval``) — the same path
the round-12 trajectory eval used. The rig's per-checkpoint
``assert_adapter_actually_applied`` guard fires immediately on any recurrence
of the #477 v4/v6 silent-LoRA-not-applied regression, so we never silently
write ΔG ≈ 0 to disk again.

NOT TRAINING. Re-eval only. Two cells only (r=4 + r=8).

Per cell the driver:
  1. Resolves the checkpoint index from
     ``/workspace/runs/issue_504/<cell>_seed<seed>/checkpoint_index.json``
     (built by ``i504_run_cell.py`` during training).
  2. Resolves the held-out panel from the Phase 0.5 output
     ``eval_results/issue_504/phase0_5_gates_round6.json``.
  3. Asserts the disjointness guard (panel ∩ negatives must be empty).
  4. Calls ``run_trajectory_eval`` — the SAME production primitive
     (vLLM LoRARequest + score_logp_for_R + eval_guard).
  5. Persists per-cell to
     ``eval_results/issue_504/reval_grid/<cell>_seed<seed>/trajectory.json``
     (the rig's natural output path; we point it there directly).
  6. Idempotent resume: cells whose ``trajectory.json`` already exists are
     SKIPPED (the rig itself is checkpoint-per-frac internally for crash
     safety, but a fully-landed trajectory.json means the cell is done).

After both cells: aggregate to
``eval_results/issue_504/reval_grid/grid.json``.

Parallelism (``--gpus N``): the driver partitions the 2-cell list N ways and
spawns N WORKER subprocesses with ``CUDA_VISIBLE_DEVICES=k``. Each worker
processes its slice sequentially in-process. With ``--gpus 4`` the 2 cells go
1+1 with 2 idle GPUs (fine — saves wall-time on hard-coupled vLLM teardown).

Subprocess env is explicit (``env={**os.environ, "CUDA_VISIBLE_DEVICES": str(k)}``)
to satisfy the CLAUDE.md subprocess-env-explicit contract; ``load_dotenv()``
at module-top ensures HF_TOKEN / WANDB_API_KEY land in the parent's env before
any subprocess copies it.

Launch:
    uv run python scripts/i504_reval_grid.py --gpus 4 --max-new-tokens 1024
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

# Two-check subprocess-env-passthrough contract: load_dotenv() at module-top so
# HF_TOKEN + WANDB_API_KEY land in the parent's env BEFORE any subprocess copies
# it (this script spawns workers with env={**os.environ, CVD=k}).
load_dotenv()

log = logging.getLogger("i504.reval_grid")

# ── Constants ───────────────────────────────────────────────────────────────
ADAPTER_HF_REPO = "superkaiba1/explore-persona-space"
ADAPTER_SUBFOLDER_ROOT = "adapters/issue_504"

DEFAULT_RUNS_ROOT = Path("/workspace/runs/issue_504")
DEFAULT_OUT_ROOT = Path("eval_results/issue_504/reval_grid")
DEFAULT_PANEL_JSON = Path("eval_results/issue_504/phase0_5_gates_round6.json")
DEFAULT_BANK_PATH = Path("data/issue_472/persona_bank.json")

DEFAULT_MAX_NEW_TOKENS = 1024  # ≥ 2× trained completion (CLAUDE.md)
DEFAULT_GPU_MEM_UTIL = 0.60

# The 2 cells with adapters on disk after round-12 training.
DEFAULT_CELLS: tuple[tuple[str, int], ...] = (
    ("c504_smoke_r4", 42),
    ("c504_smoke_r8", 42),
)

# Map cell slug → rank for max_lora_rank threading. (Smoke slugs encode their
# rank in the name.)
_CELL_RANK_MAP: dict[str, int] = {
    "c504_smoke_r4": 4,
    "c504_smoke_r8": 8,
    "c504_smoke_r16": 16,
}


@dataclass(frozen=True)
class CellEntry:
    """One row in the 2-cell re-eval grid."""

    cell: str
    seed: int
    rank: int  # adapter's actual rank; vLLM max_lora_rank is floored to max(8, rank)

    @property
    def run_dirname(self) -> str:
        return f"{self.cell}_seed{self.seed}"


def _rank_for_cell(cell: str) -> int:
    """Return the adapter rank for the given smoke cell slug.

    Raises:
        KeyError: cell slug not in the known map (only c504_smoke_r{4,8,16}
            currently appear in round-12 training).
    """
    if cell not in _CELL_RANK_MAP:
        raise KeyError(
            f"cell {cell!r} not in known #504 smoke-rank map {sorted(_CELL_RANK_MAP)}; "
            "extend _CELL_RANK_MAP if a new smoke cell was added."
        )
    return _CELL_RANK_MAP[cell]


def _git_sha() -> str:
    """Best-effort git HEAD sha; 'unknown' on failure."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            # epm-lint: subprocess-env-inherit -- git rev-parse needs no credentials
            env={**os.environ},
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _env_versions() -> dict[str, str]:
    """Pinned env versions for reproducibility metadata."""
    versions: dict[str, str] = {}
    for pkg in ("vllm", "peft", "transformers", "torch"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"
    return versions


def _load_panel(panel_json: Path) -> tuple[list[str], str, dict[str, str], str | None]:
    """Read the Phase 0.5 panel JSON.

    Returns (held_out_panel, default_persona, arm_to_positioned_n, smoke_mid_band_n).
    Raises if the file lacks the load-bearing keys.
    """
    payload = json.loads(panel_json.read_text())
    held_out_panel = payload.get("held_out_panel", [])
    if not held_out_panel:
        raise RuntimeError(
            f"--panel-json {panel_json} has empty 'held_out_panel' — Phase 0.5 must "
            "have run before this driver."
        )
    chosen_negatives = payload.get("chosen_negatives", {})
    default_persona = chosen_negatives.get("default", "qwen_default")
    arm_to_positioned_n = payload.get("arm_to_positioned_n", {})
    smoke_mid_band_n = payload.get("smoke_mid_band_n")
    return held_out_panel, default_persona, arm_to_positioned_n, smoke_mid_band_n


def _cell_negatives(
    entry: CellEntry,
    default_persona: str,
    arm_to_positioned_n: dict[str, str],
    smoke_mid_band_n: str | None,
) -> set[str]:
    """Compute the set of personas the cell trained against (for the disjointness
    guard). Smoke cells use ``smoke_mid_band_n``; positioned cells use
    ``arm_to_positioned_n[cell]``; default-only carries no positioned-N."""
    negs: set[str] = {default_persona}
    # v1-only recovery rig — c504_smoke_r{4,8}_seed42 only. The CELL_TO_RANK
    # map above (~line 92) lists ONLY the v1 r-ladder slugs; this driver is
    # a round-12 recovery rig over those exact two LoRAs and is NEVER fed
    # v2/v3 smoke cells. Do NOT add c504v2_smoke_/c504v3_smoke_ here — the
    # active eval-trajectory path lives at scripts/i504_eval_trajectory.py
    # and its disjointness guard already covers v1/v2/v3 (round-7 fix).
    if entry.cell.startswith("c504_smoke_"):  # epm-smoke-prefix: v1-only-recovery
        if smoke_mid_band_n is None:
            raise RuntimeError(
                f"cell {entry.cell!r} is a smoke cell but panel JSON has no "
                "'smoke_mid_band_n' key; Phase 0.5 must populate it."
            )
        negs.add(smoke_mid_band_n)
    elif entry.cell in arm_to_positioned_n:
        negs.add(arm_to_positioned_n[entry.cell])
    return negs


def _load_checkpoint_specs(run_dir: Path) -> list[dict]:
    """Load the checkpoint_index.json built by training; convert to the rig's
    checkpoint_specs shape (frac/step/adapter_path)."""
    idx_path = run_dir / "checkpoint_index.json"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"checkpoint_index.json missing at {idx_path}; "
            "training did not complete or the run directory is wrong."
        )
    idx = json.loads(idx_path.read_text())
    specs: list[dict] = []
    for frac_str, entry in sorted(idx.items(), key=lambda kv: float(kv[0])):
        if entry.get("path") is None:
            log.warning(
                "[%s] checkpoint frac=%s has no 'path'; skipping (training may have "
                "skipped this fraction).",
                run_dir.name,
                frac_str,
            )
            continue
        specs.append(
            {
                "frac": float(frac_str),
                "step": entry.get("step"),
                "adapter_path": entry["path"],
            }
        )
    if not specs:
        raise RuntimeError(
            f"checkpoint_index at {idx_path} has zero usable checkpoints — "
            "training may have written nothing."
        )
    return specs


def _eval_one_cell(
    *,
    entry: CellEntry,
    runs_root: Path,
    out_root: Path,
    panel_json: Path,
    bank_path: Path,
    max_new_tokens: int,
    gpu_mem_util: float,
    no_kl: bool,
) -> Path:
    """Re-eval ONE cell end-to-end via run_trajectory_eval. Idempotent.

    Skips silently if the per-cell output (``trajectory.json``) already exists.
    Returns the per-cell trajectory.json path.
    """
    cell_out_dir = out_root / entry.run_dirname
    cell_out_path = cell_out_dir / "trajectory.json"
    if cell_out_path.exists():
        log.info(
            "[%s] per-cell trajectory.json exists — skipping (idempotent resume): %s",
            entry.run_dirname,
            cell_out_path,
        )
        return cell_out_path

    # ── Phase 0: panel + bank + checkpoint specs. ────────────────────────────
    held_out_panel, default_persona, arm_to_positioned_n, smoke_mid_band_n = _load_panel(panel_json)
    cell_negs = _cell_negatives(entry, default_persona, arm_to_positioned_n, smoke_mid_band_n)
    overlap = set(held_out_panel) & cell_negs
    if overlap:
        raise AssertionError(
            f"panel ∩ negatives for cell={entry.cell!r}: {sorted(overlap)} — "
            "bystander ΔG would reflect training-against, not leakage."
        )
    log.info(
        "[%s] disjoint guard PASS: panel=%d personas, negs=%s",
        entry.run_dirname,
        len(held_out_panel),
        sorted(cell_negs),
    )

    run_dir = runs_root / entry.run_dirname
    checkpoint_specs = _load_checkpoint_specs(run_dir)
    log.info(
        "[%s] %d checkpoints to eval: %s",
        entry.run_dirname,
        len(checkpoint_specs),
        [c["frac"] for c in checkpoint_specs],
    )

    # ── Phase 1: load bank + q_eval + call run_trajectory_eval. ──────────────
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        SOURCE_PERSONA,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        run_trajectory_eval,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        get_train_eval_questions,
    )

    bank = load_persona_bank(bank_path)
    for p in held_out_panel:
        if p not in bank:
            raise KeyError(
                f"Panel persona {p!r} missing from bank at {bank_path}; "
                "Phase 0.5 and the bank must be the SAME artifact."
            )
    eval_personas = {p: bank[p] for p in held_out_panel}
    _q_train, q_eval = get_train_eval_questions()

    # vLLM's max_lora_rank is a buffer size in {8, 16, 32, 64, 128, 256, 320, 512};
    # r=4 fits in an r=8 buffer (zero-padded). The production rig
    # (i504_eval_trajectory.py) does the SAME floor — mirror it here so the
    # measurement is byte-identical.
    vllm_max_lora_rank = max(8, entry.rank)
    log.info(
        "[%s] max_lora_rank: training=%d → vLLM buffer=%d",
        entry.run_dirname,
        entry.rank,
        vllm_max_lora_rank,
    )

    cell_out_dir.mkdir(parents=True, exist_ok=True)
    run_trajectory_eval(
        cell_slug=entry.cell,
        seed=entry.seed,
        checkpoint_specs=checkpoint_specs,
        eval_personas=eval_personas,
        eval_questions=q_eval,
        source=SOURCE_PERSONA,
        source_prompt=bank[SOURCE_PERSONA],
        out_path=cell_out_path,
        max_new_tokens=max_new_tokens,
        max_lora_rank=vllm_max_lora_rank,
        gpu_memory_utilization=gpu_mem_util,
        compute_kl=not no_kl,
    )

    if not cell_out_path.exists():
        raise RuntimeError(
            f"run_trajectory_eval exited but {cell_out_path} is missing — silent failure."
        )
    log.info("[%s] DONE → %s", entry.run_dirname, cell_out_path)
    return cell_out_path


def _aggregate_grid(out_root: Path, entries: list[CellEntry]) -> Path:
    """Walk every per-cell trajectory.json and stitch into grid.json.

    Extracts the final-checkpoint summary per cell (the matched-slice anchor
    + source-self ΔG + bystander stats) so the grid table is self-contained
    without loading 6 nested trajectories per cell.
    """
    rows: list[dict] = []
    missing: list[str] = []
    for entry in entries:
        cell_out = out_root / entry.run_dirname / "trajectory.json"
        if not cell_out.exists():
            missing.append(entry.run_dirname)
            continue
        payload = json.loads(cell_out.read_text())
        checkpoints = payload.get("checkpoints", [])
        # The final checkpoint (frac = 1.0) is the matched-slice anchor.
        terminal = next(
            (c for c in checkpoints if c.get("frac") == 1.0),
            checkpoints[-1] if checkpoints else None,
        )
        if terminal is None:
            log.warning("[%s] trajectory.json has zero checkpoints", entry.run_dirname)
            continue
        source_self = terminal.get("source_self", {})
        # Bystander summary: mean ΔG + emit rate across held-out panel at frac=terminal.
        held_out = terminal.get("held_out", {})
        all_dgs: list[float] = []
        all_emits: list[bool] = []
        for _persona, per_q in held_out.items():
            for _q, leaf in per_q.items():
                all_dgs.append(float(leaf.get("delta_g", 0.0)))
                all_emits.append(bool(leaf.get("argmax_marker", False)))
        held_out_delta_g_mean = sum(all_dgs) / len(all_dgs) if all_dgs else float("nan")
        held_out_emit_rate = (
            sum(1 for x in all_emits if x) / len(all_emits) if all_emits else float("nan")
        )
        rows.append(
            {
                "cell": entry.cell,
                "seed": entry.seed,
                "rank": entry.rank,
                "trajectory_path": str(cell_out),
                "terminal_frac": terminal.get("frac"),
                "source_self_delta_g_mean": source_self.get("delta_g_mean"),
                "source_emit_rate": source_self.get("emission_p"),
                "held_out_delta_g_mean": held_out_delta_g_mean,
                "held_out_emit_rate": held_out_emit_rate,
                "n_held_out_collapsed": terminal.get("n_held_out_collapsed", 0),
                "held_out_collapse_share": terminal.get("held_out_collapse_share", 0.0),
                "n_checkpoints": len(checkpoints),
            }
        )

    grid_path = out_root / "grid.json"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path.write_text(
        json.dumps(
            {
                "schema_version": "i504_reval_grid_v1",
                "n_cells_total": len(entries),
                "n_cells_persisted": len(rows),
                "n_cells_missing": len(missing),
                "missing_cells": missing,
                "rows": rows,
                "git_commit": _git_sha(),
                "hostname": socket.gethostname(),
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "env": _env_versions(),
            },
            indent=2,
        )
    )
    log.info(
        "Aggregated %d/%d cells → %s (%d missing: %s)",
        len(rows),
        len(entries),
        grid_path,
        len(missing),
        missing,
    )
    return grid_path


def _run_worker_in_process(
    *,
    worker_entries: list[CellEntry],
    runs_root: Path,
    out_root: Path,
    panel_json: Path,
    bank_path: Path,
    max_new_tokens: int,
    gpu_mem_util: float,
    no_kl: bool,
) -> int:
    """Eval the assigned entries sequentially in-process (single-GPU path or
    inside a worker subprocess). Failures RAISE; checkpoint-per-frac inside
    ``run_trajectory_eval`` guarantees earlier work is on disk."""
    for entry in worker_entries:
        _eval_one_cell(
            entry=entry,
            runs_root=runs_root,
            out_root=out_root,
            panel_json=panel_json,
            bank_path=bank_path,
            max_new_tokens=max_new_tokens,
            gpu_mem_util=gpu_mem_util,
            no_kl=no_kl,
        )
    return 0


def _partition(entries: list[CellEntry], n_gpus: int) -> list[list[CellEntry]]:
    """Round-robin entries across GPU slices. With 2 entries and 4 GPUs, the
    first 2 slices get one entry each, the last 2 are empty (skipped at spawn)."""
    return [entries[i::n_gpus] for i in range(n_gpus)]


def _spawn_worker_subprocesses(
    *,
    partitions: list[list[CellEntry]],
    runs_root: Path,
    out_root: Path,
    panel_json: Path,
    bank_path: Path,
    max_new_tokens: int,
    gpu_mem_util: float,
    no_kl: bool,
    script_path: Path,
) -> int:
    """Spawn one worker subprocess per non-empty partition with CUDA_VISIBLE_DEVICES=k."""
    procs: list[tuple[int, subprocess.Popen]] = []
    for gpu_id, slice_ in enumerate(partitions):
        if not slice_:
            log.info("[gpu=%d] partition empty — no worker spawned", gpu_id)
            continue
        worker_cells = ",".join(f"{e.cell}:{e.seed}" for e in slice_)
        # Explicit env passthrough: {**os.environ} + CVD override; load_dotenv()
        # at module-top guarantees HF_TOKEN already lives in os.environ.
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        cmd = [
            "uv",
            "run",
            "python",
            str(script_path),
            "--worker-cells",
            worker_cells,
            "--runs-root",
            str(runs_root),
            "--out-root",
            str(out_root),
            "--panel-json",
            str(panel_json),
            "--bank-path",
            str(bank_path),
            "--max-new-tokens",
            str(max_new_tokens),
            "--gpu-mem-util",
            str(gpu_mem_util),
        ]
        if no_kl:
            cmd.append("--no-kl")
        log.info("[gpu=%d] spawning worker on %d cells: %s", gpu_id, len(slice_), worker_cells)
        p = subprocess.Popen(cmd, env=env)
        procs.append((gpu_id, p))

    failures: list[tuple[int, int]] = []
    for gpu_id, p in procs:
        rc = p.wait()
        if rc != 0:
            failures.append((gpu_id, rc))
            log.error("[gpu=%d] worker exited rc=%d", gpu_id, rc)
        else:
            log.info("[gpu=%d] worker exited rc=0", gpu_id)
    if failures:
        log.error("%d worker(s) failed: %s", len(failures), failures)
        return 1
    return 0


def _parse_worker_cells(spec: str) -> list[CellEntry]:
    """Parse '--worker-cells cell:seed,cell:seed' into a list of CellEntry."""
    out: list[CellEntry] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(
                f"--worker-cells entry {raw!r} must be in 'cell:seed' form, e.g. "
                "'c504_smoke_r4:42'."
            )
        cell, seed_str = raw.split(":", 1)
        out.append(CellEntry(cell=cell, seed=int(seed_str), rank=_rank_for_cell(cell)))
    return out


def _build_argparser() -> argparse.ArgumentParser:
    """CLI parser. Extracted so main() stays under McCabe 15."""
    ap = argparse.ArgumentParser(
        description=(
            "Task #504 recovery re-eval driver — recover the leakage grid via the "
            "vLLM-LoRARequest path on the current env (post-eval-guard)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to parallelize across (spawns N worker subprocesses, "
        "one per GPU, with CUDA_VISIBLE_DEVICES=k). With 2 cells and --gpus 4 the "
        "partition is 1/1/0/0 — 2 idle GPUs is OK.",
    )
    ap.add_argument(
        "--cells",
        default=None,
        help="Optional comma-separated list of 'cell:seed' pairs to eval. Default = "
        "all 2 round-12-trained cells (c504_smoke_r4:42, c504_smoke_r8:42).",
    )
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--gpu-mem-util", type=float, default=DEFAULT_GPU_MEM_UTIL)
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--panel-json", type=Path, default=DEFAULT_PANEL_JSON)
    ap.add_argument("--bank-path", type=Path, default=DEFAULT_BANK_PATH)
    ap.add_argument(
        "--no-kl",
        action="store_true",
        help="Skip DV-B full-vocab KL (saves ~30% wall on the eval rig — diagnostic "
        "still recovers the marker log-prob signal).",
    )
    ap.add_argument(
        "--worker-cells",
        default=None,
        help="Internal: comma-separated 'cell:seed' list this in-process worker owns. "
        "When set, --gpus is ignored and we eval the listed entries in-process.",
    )
    ap.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip the final grid.json aggregation (default off; aggregation is fast).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the cell list + per-GPU partition and exit (no fetch, no eval). "
        "Used for the dev-VM smoke check.",
    )
    return ap


def _select_entries(cells_arg: str | None) -> list[CellEntry]:
    """Build the CellEntry list from --cells or the DEFAULT_CELLS constant."""
    if cells_arg:
        entries: list[CellEntry] = []
        for raw in cells_arg.split(","):
            raw = raw.strip()
            if ":" not in raw:
                raise ValueError(
                    f"--cells entry {raw!r} must be in 'cell:seed' form, e.g. 'c504_smoke_r4:42'."
                )
            cell, seed_str = raw.split(":", 1)
            entries.append(CellEntry(cell=cell, seed=int(seed_str), rank=_rank_for_cell(cell)))
        return entries
    return [
        CellEntry(cell=cell, seed=seed, rank=_rank_for_cell(cell)) for cell, seed in DEFAULT_CELLS
    ]


def main(argv: list[str] | None = None) -> int:
    ap = _build_argparser()
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=reval_grid] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    token = os.environ.get("HF_TOKEN")
    if token is None and not args.dry_run:
        raise RuntimeError(
            "HF_TOKEN missing — load_dotenv() ran but .env lacks the token; the "
            "bank load + adapter access need it. Fix .env on the pod."
        )

    entries = _select_entries(args.cells)
    log.info("entries: %s", [e.run_dirname for e in entries])

    # Worker branch — in-process eval of the assigned entries.
    if args.worker_cells is not None:
        worker_entries = _parse_worker_cells(args.worker_cells)
        log.info("worker: %d entries assigned", len(worker_entries))
        return _run_worker_in_process(
            worker_entries=worker_entries,
            runs_root=args.runs_root,
            out_root=args.out_root,
            panel_json=args.panel_json,
            bank_path=args.bank_path,
            max_new_tokens=args.max_new_tokens,
            gpu_mem_util=args.gpu_mem_util,
            no_kl=args.no_kl,
        )

    # Driver branch — partition + dispatch.
    partitions = _partition(entries, args.gpus)
    log.info(
        "partitioned %d entries across %d GPU slices: sizes=%s",
        len(entries),
        args.gpus,
        [len(p) for p in partitions],
    )
    if args.dry_run:
        print("\n[dry-run] cell partition")
        for gpu_id, slice_ in enumerate(partitions):
            if not slice_:
                print(f"  [gpu={gpu_id}] EMPTY (would not spawn worker)")
                continue
            print(f"  [gpu={gpu_id}] {len(slice_)} cells:")
            for entry in slice_:
                print(f"    {entry.run_dirname}  (rank={entry.rank})")
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "verdict": "DRY_RUN_PASS",
                    "n_entries": len(entries),
                    "n_gpus": args.gpus,
                    "partition_sizes": [len(p) for p in partitions],
                    "runs_root": str(args.runs_root),
                    "out_root": str(args.out_root),
                    "panel_json": str(args.panel_json),
                    "panel_json_exists": args.panel_json.exists(),
                    "max_new_tokens": args.max_new_tokens,
                    "env": _env_versions(),
                },
                indent=2,
            )
        )
        return 0

    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.gpus == 1:
        rc = _run_worker_in_process(
            worker_entries=entries,
            runs_root=args.runs_root,
            out_root=args.out_root,
            panel_json=args.panel_json,
            bank_path=args.bank_path,
            max_new_tokens=args.max_new_tokens,
            gpu_mem_util=args.gpu_mem_util,
            no_kl=args.no_kl,
        )
    else:
        script_path = Path(__file__).resolve()
        rc = _spawn_worker_subprocesses(
            partitions=partitions,
            runs_root=args.runs_root,
            out_root=args.out_root,
            panel_json=args.panel_json,
            bank_path=args.bank_path,
            max_new_tokens=args.max_new_tokens,
            gpu_mem_util=args.gpu_mem_util,
            no_kl=args.no_kl,
            script_path=script_path,
        )

    if not args.no_aggregate:
        grid_path = _aggregate_grid(args.out_root, entries)
        log.info("grid.json → %s", grid_path)
    return rc


if __name__ == "__main__":
    sys.exit(main())
