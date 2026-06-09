# ruff: noqa: RUF003  # em-dash + Qwen marker " ※" + Greek ΔG + minus sign − intentional
#!/usr/bin/env python3
"""Task #530 inline follow-up (`logit_reval`) — re-eval the 10 final adapters with
the logit-instrumented eval rig.

The published #530 run stored only ``g_logp``/``b_logp``/``kl`` per (probe, q)
pair; raw marker-slot logits (``z_marker``, ``z_eos``, ``logZ``) were never
captured, and vLLM's logprobs API cannot recover them (post-softmax only). This
driver re-runs the production #472 eval rig (``run_trajectory_eval``), whose
Phase B now captures the raw logits from the SAME HF teacher-forced forward
pass as the KL — see ``.claude/rules/marker-leakage-measurement.md`` § "Report
BOTH log-prob and logit".

NOT TRAINING. Re-eval only. 10 cells: ``c504v3_{near,mid_near,mid_far,far,
default_only}_seed{42,137}``, each at its band-stop final checkpoint (step 20,
frac 1.0), adapters pulled from HF ``superkaiba1/explore-persona-space`` under
``adapters/issue_530/<cell>_seed<seed>/``.

Per cell the driver:
  1. Downloads the top-level adapter files (``adapter_config.json`` +
     ``adapter_model.safetensors``) via ``HfApi.list_repo_files`` + per-file
     ``hf_hub_download``. NOT ``snapshot_download(allow_patterns=...)`` — on
     this >8k-file repo the ``repo_info.siblings`` listing truncates and
     ``snapshot_download`` silently returns 0 files (#375, #399 incidents).
  2. Resolves the held-out panel from ``eval_results/issue_530/
     phase0_5_gates.json`` + the persona bank, and Q_eval from the published
     ``c504v3_near_seed42/trajectory.json`` (the SAME 10 questions).
  3. Asserts the disjointness guard (panel ∩ cell negatives must be empty).
  4. Calls ``run_trajectory_eval`` with ``checkpoint_specs=[{"frac": 1.0,
     "step": 20, "adapter_path": <local>}]``, ``compute_kl=True``,
     ``max_new_tokens=2048``. The rig's gauge assert
     (``assert_logit_readout_gauge_free``) fails loud if any adapter
     LoRA-touches the unembedding (it would invalidate the Δz_marker readout).
  5. Persists per-cell to
     ``eval_results/issue_530/logit_reval/<cell>_seed<seed>/trajectory.json``
     (the slug layout ``analyze.load_trajectory`` expects, so
     ``scripts/issue530_logit_analysis.py`` reads the slab directly).
  6. Idempotent resume: cells whose ``trajectory.json`` exists are SKIPPED.

After all cells: aggregate to ``eval_results/issue_530/logit_reval/grid.json``.

Parallelism (``--gpus N``): partitions the 10-cell list N ways and spawns N
WORKER subprocesses with ``CUDA_VISIBLE_DEVICES=k`` (same dispatcher shape as
``scripts/i504_reval_grid.py``; smoke = this same script with ``--dry-run`` or
``--cells <one cell> --gpus 1`` — no separate smoke path).

Launch:
    uv run python scripts/issue530_logit_reval.py --gpus 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

# Two-check subprocess-env-passthrough contract: load_dotenv() at module-top so
# HF_TOKEN + WANDB_API_KEY land in the parent's env BEFORE any subprocess copies
# it (this script spawns workers with env={**os.environ, CVD=k}).
load_dotenv()

log = logging.getLogger("i530.logit_reval")

# ── Constants ───────────────────────────────────────────────────────────────
ADAPTER_HF_REPO = "superkaiba1/explore-persona-space"
ADAPTER_SUBFOLDER_ROOT = "adapters/issue_530"

DEFAULT_OUT_ROOT = Path("eval_results/issue_530/logit_reval")
DEFAULT_PANEL_JSON = Path("eval_results/issue_530/phase0_5_gates.json")
DEFAULT_BANK_PATH = Path("data/issue_472/persona_bank.json")
DEFAULT_QUESTIONS_TRAJ = Path("eval_results/issue_530/c504v3_near_seed42/trajectory.json")
DEFAULT_ADAPTER_CACHE = Path("/workspace/runs/issue_530/logit_reval_adapters")

# Marker / end-of-completion eval: ≥ 2× longest trained completion (CLAUDE.md);
# the original #530 run used 2048.
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_GPU_MEM_UTIL = 0.60

# All 10 #530 cells trained at LoRA r=8 (body Reproducibility table); vLLM's
# max_lora_rank is a buffer size with minimum 8, so the floor is a no-op here.
LORA_RANK = 8
VLLM_MAX_LORA_RANK = max(8, LORA_RANK)

# Band-stop final checkpoint for every cell (the only one trained — the stop
# fired at the FIRST eval boundary, step 20; intermediate fractions were never
# reached, per the #530 body's Artifacts note).
FINAL_CHECKPOINT_FRAC = 1.0
FINAL_CHECKPOINT_STEP = 20

ARMS: tuple[str, ...] = ("near", "mid_near", "mid_far", "far", "default_only")
SEEDS: tuple[int, ...] = (42, 137)
DEFAULT_CELLS: tuple[tuple[str, int], ...] = tuple(
    (f"c504v3_{arm}", seed) for arm in ARMS for seed in SEEDS
)

REQUIRED_ADAPTER_FILES: tuple[str, ...] = ("adapter_config.json", "adapter_model.safetensors")


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


def _run_dirname(cell: str, seed: int) -> str:
    return f"{cell}_seed{seed}"


def _load_panel(panel_json: Path) -> tuple[list[str], str, dict[str, str]]:
    """Read the Phase 0.5 panel JSON → (held_out_panel, default_persona, arm_to_positioned_n)."""
    payload = json.loads(panel_json.read_text())
    held_out_panel = payload.get("held_out_panel", [])
    if not held_out_panel:
        raise RuntimeError(
            f"--panel-json {panel_json} has empty 'held_out_panel' — the #530 Phase 0.5 "
            "artifact is required for this re-eval."
        )
    chosen_negatives = payload.get("chosen_negatives", {})
    default_persona = chosen_negatives.get("default", "qwen_default")
    arm_to_positioned_n = payload.get("arm_to_positioned_n", {})
    return held_out_panel, default_persona, arm_to_positioned_n


def _load_eval_questions(questions_traj: Path) -> list[str]:
    """Read Q_eval from the published #530 trajectory (the SAME 10 questions)."""
    payload = json.loads(questions_traj.read_text())
    qs = payload.get("eval_questions", [])
    if not qs:
        raise RuntimeError(
            f"--questions-traj {questions_traj} has empty 'eval_questions' — cannot "
            "reproduce the published Q_eval."
        )
    log.info("Q_eval: %d questions from %s", len(qs), questions_traj)
    return qs


def _cell_negatives(cell: str, default_persona: str, arm_to_positioned_n: dict[str, str]) -> set:
    """Personas the cell trained against (for the panel-disjointness guard)."""
    negs = {default_persona}
    if cell in arm_to_positioned_n:
        negs.add(arm_to_positioned_n[cell])
    elif cell != "c504v3_default_only":
        raise KeyError(
            f"cell {cell!r} has no positioned-N entry in arm_to_positioned_n and is not "
            "the default-only arm — unknown cell; refusing to guess its negative set."
        )
    return negs


def _download_adapter(cell: str, seed: int, cache_root: Path, repo_files: list[str]) -> Path:
    """Fetch the top-level adapter files for one cell from the HF model repo.

    Per-file ``hf_hub_download`` against an explicit ``list_repo_files``
    listing (NOT ``snapshot_download(allow_patterns=...)`` — silently empty on
    this repo's truncated siblings, #375/#399). Idempotent: returns early when
    both required files already exist locally.

    Returns the local adapter directory (contains adapter_config.json +
    adapter_model.safetensors).
    """
    from huggingface_hub import hf_hub_download

    slug = _run_dirname(cell, seed)
    prefix = f"{ADAPTER_SUBFOLDER_ROOT}/{slug}/"
    adapter_dir = cache_root / ADAPTER_SUBFOLDER_ROOT / slug
    if all((adapter_dir / f).exists() for f in REQUIRED_ADAPTER_FILES):
        log.info("[%s] adapter already cached at %s — skipping download", slug, adapter_dir)
        return adapter_dir

    top_level = [f for f in repo_files if f.startswith(prefix) and "/" not in f[len(prefix) :]]
    required = {prefix + f for f in REQUIRED_ADAPTER_FILES}
    missing = sorted(required - set(top_level))
    if missing:
        nearby = sorted(f for f in repo_files if f.startswith(prefix))[:10]
        raise FileNotFoundError(
            f"[{slug}] required adapter files missing on HF repo {ADAPTER_HF_REPO}: "
            f"{missing}. Files under the prefix (first 10): {nearby}. The #530 body "
            "records these adapters as uploaded — repo drift; do NOT fall back to "
            "retraining without checking."
        )
    for fname in sorted(set(top_level)):
        local = hf_hub_download(
            repo_id=ADAPTER_HF_REPO,
            filename=fname,
            local_dir=cache_root,
            token=os.environ.get("HF_TOKEN"),
        )
        log.info("[%s] downloaded %s → %s", slug, fname, local)
    for f in REQUIRED_ADAPTER_FILES:
        if not (adapter_dir / f).exists():
            raise RuntimeError(
                f"[{slug}] post-download invariant failed: {adapter_dir / f} missing "
                "even though the Hub listing contained it — hf_hub_download local_dir "
                "layout drift; inspect the cache root."
            )
    return adapter_dir


def _eval_one_cell(
    *,
    cell: str,
    seed: int,
    out_root: Path,
    panel_json: Path,
    bank_path: Path,
    questions_traj: Path,
    adapter_cache: Path,
    max_new_tokens: int,
    gpu_mem_util: float,
) -> Path:
    """Re-eval ONE cell end-to-end via run_trajectory_eval. Idempotent."""
    slug = _run_dirname(cell, seed)
    cell_out_path = out_root / slug / "trajectory.json"
    if cell_out_path.exists():
        log.info("[%s] trajectory.json exists — skipping (idempotent resume)", slug)
        return cell_out_path

    held_out_panel, default_persona, arm_to_positioned_n = _load_panel(panel_json)
    cell_negs = _cell_negatives(cell, default_persona, arm_to_positioned_n)
    overlap = set(held_out_panel) & cell_negs
    if overlap:
        raise AssertionError(
            f"panel ∩ negatives for cell={cell!r}: {sorted(overlap)} — bystander ΔG "
            "would reflect training-against, not leakage."
        )
    log.info(
        "[%s] disjoint guard PASS: panel=%d personas, negs=%s",
        slug,
        len(held_out_panel),
        sorted(cell_negs),
    )

    from huggingface_hub import list_repo_files

    repo_files = list_repo_files(ADAPTER_HF_REPO, token=os.environ.get("HF_TOKEN"))
    adapter_dir = _download_adapter(cell, seed, adapter_cache, repo_files)

    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        SOURCE_PERSONA,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        run_trajectory_eval,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )

    bank = load_persona_bank(bank_path)
    for p in held_out_panel:
        if p not in bank:
            raise KeyError(
                f"Panel persona {p!r} missing from bank at {bank_path}; the panel and "
                "the bank must be the SAME artifacts the #530 run used."
            )
    eval_personas = {p: bank[p] for p in held_out_panel}
    eval_questions = _load_eval_questions(questions_traj)

    checkpoint_specs = [
        {
            "frac": FINAL_CHECKPOINT_FRAC,
            "step": FINAL_CHECKPOINT_STEP,
            "adapter_path": str(adapter_dir),
        }
    ]

    cell_out_path.parent.mkdir(parents=True, exist_ok=True)
    run_trajectory_eval(
        cell_slug=cell,
        seed=seed,
        checkpoint_specs=checkpoint_specs,
        eval_personas=eval_personas,
        eval_questions=eval_questions,
        source=SOURCE_PERSONA,
        source_prompt=bank[SOURCE_PERSONA],
        out_path=cell_out_path,
        max_new_tokens=max_new_tokens,
        max_lora_rank=VLLM_MAX_LORA_RANK,
        gpu_memory_utilization=gpu_mem_util,
        compute_kl=True,  # the logit capture rides Phase B — never skip it here
    )

    if not cell_out_path.exists():
        raise RuntimeError(
            f"run_trajectory_eval exited but {cell_out_path} is missing — silent failure."
        )
    log.info("[%s] DONE → %s", slug, cell_out_path)
    return cell_out_path


def _aggregate_grid(out_root: Path, entries: list[tuple[str, int]]) -> Path:
    """Stitch per-cell trajectories into grid.json with logit-space summaries."""
    rows: list[dict] = []
    missing: list[str] = []
    for cell, seed in entries:
        slug = _run_dirname(cell, seed)
        cell_out = out_root / slug / "trajectory.json"
        if not cell_out.exists():
            missing.append(slug)
            continue
        payload = json.loads(cell_out.read_text())
        checkpoints = payload.get("checkpoints", [])
        terminal = next(
            (c for c in checkpoints if c.get("frac") == FINAL_CHECKPOINT_FRAC),
            checkpoints[-1] if checkpoints else None,
        )
        if terminal is None:
            log.warning("[%s] trajectory.json has zero checkpoints", slug)
            continue
        source_self = terminal.get("source_self", {})
        held_out = terminal.get("held_out", {})
        dgs: list[float] = []
        dzs: list[float] = []
        dmargins: list[float] = []
        hf_vllm_gaps: list[float] = []
        for per_q in held_out.values():
            for leaf in per_q.values():
                dgs.append(float(leaf["delta_g"]))
                if leaf.get("delta_z_marker") is not None:
                    dzs.append(float(leaf["delta_z_marker"]))
                if leaf.get("delta_margin") is not None:
                    dmargins.append(float(leaf["delta_margin"]))
                if leaf.get("logp_hf_g") is not None:
                    hf_vllm_gaps.append(abs(float(leaf["logp_hf_g"]) - float(leaf["g_logp"])))

        def _mean(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else float("nan")

        rows.append(
            {
                "cell": cell,
                "seed": seed,
                "trajectory_path": str(cell_out),
                "source_self_delta_g_mean": source_self.get("delta_g_mean"),
                "source_delta_z_marker_mean": source_self.get("delta_z_marker_mean"),
                "held_out_delta_g_mean": _mean(dgs),
                "held_out_delta_z_marker_mean": _mean(dzs),
                "held_out_delta_margin_mean": _mean(dmargins),
                "hf_vs_vllm_logp_abs_gap_mean": _mean(hf_vllm_gaps),
                "n_held_out_pairs": len(dgs),
            }
        )

    grid_path = out_root / "grid.json"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path.write_text(
        json.dumps(
            {
                "schema_version": "i530_logit_reval_v1",
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


def _run_worker_in_process(*, worker_entries: list[tuple[str, int]], args) -> int:
    """Eval the assigned entries sequentially in-process. Failures RAISE;
    checkpoint-per-frac inside run_trajectory_eval keeps earlier work on disk."""
    for cell, seed in worker_entries:
        _eval_one_cell(
            cell=cell,
            seed=seed,
            out_root=args.out_root,
            panel_json=args.panel_json,
            bank_path=args.bank_path,
            questions_traj=args.questions_traj,
            adapter_cache=args.adapter_cache,
            max_new_tokens=args.max_new_tokens,
            gpu_mem_util=args.gpu_mem_util,
        )
    return 0


def _partition(entries: list[tuple[str, int]], n_gpus: int) -> list[list[tuple[str, int]]]:
    """Round-robin entries across GPU slices (empty slices skipped at spawn)."""
    return [entries[i::n_gpus] for i in range(n_gpus)]


def _spawn_worker_subprocesses(
    *, partitions: list[list[tuple[str, int]]], args, script_path: Path
) -> int:
    """One worker subprocess per non-empty partition, CUDA_VISIBLE_DEVICES=k."""
    procs: list[tuple[int, subprocess.Popen]] = []
    for gpu_id, slice_ in enumerate(partitions):
        if not slice_:
            log.info("[gpu=%d] partition empty — no worker spawned", gpu_id)
            continue
        worker_cells = ",".join(f"{cell}:{seed}" for cell, seed in slice_)
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
            "--out-root",
            str(args.out_root),
            "--panel-json",
            str(args.panel_json),
            "--bank-path",
            str(args.bank_path),
            "--questions-traj",
            str(args.questions_traj),
            "--adapter-cache",
            str(args.adapter_cache),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--gpu-mem-util",
            str(args.gpu_mem_util),
        ]
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


def _parse_cells(spec: str) -> list[tuple[str, int]]:
    """Parse 'cell:seed,cell:seed' into a list of (cell, seed)."""
    out: list[tuple[str, int]] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(
                f"cells entry {raw!r} must be in 'cell:seed' form, e.g. 'c504v3_near:42'."
            )
        cell, seed_str = raw.split(":", 1)
        out.append((cell, int(seed_str)))
    return out


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Task #530 logit_reval follow-up — re-eval the 10 band-stop final adapters "
            "with the raw-logit-instrumented #472 eval rig (log-prob + logit DVs from "
            "the same forward pass)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to parallelize across (spawns N worker subprocesses with "
        "CUDA_VISIBLE_DEVICES=k). 10 cells round-robin across the slices.",
    )
    ap.add_argument(
        "--cells",
        default=None,
        help="Optional comma-separated 'cell:seed' pairs. Default = all 10 #530 cells.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--gpu-mem-util", type=float, default=DEFAULT_GPU_MEM_UTIL)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--panel-json", type=Path, default=DEFAULT_PANEL_JSON)
    ap.add_argument("--bank-path", type=Path, default=DEFAULT_BANK_PATH)
    ap.add_argument("--questions-traj", type=Path, default=DEFAULT_QUESTIONS_TRAJ)
    ap.add_argument("--adapter-cache", type=Path, default=DEFAULT_ADAPTER_CACHE)
    ap.add_argument(
        "--worker-cells",
        default=None,
        help="Internal: 'cell:seed,...' list this worker owns; evals in-process.",
    )
    ap.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip the final grid.json aggregation.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the cell list + per-GPU partition and exit (no fetch, no eval). "
        "Dev-VM smoke check — same dispatcher path as the real run.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_argparser()
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=logit_reval] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    token = os.environ.get("HF_TOKEN")
    if token is None and not args.dry_run:
        raise RuntimeError(
            "HF_TOKEN missing — load_dotenv() ran but .env lacks the token; the bank "
            "load + adapter downloads need it. Fix .env on the pod."
        )

    entries = _parse_cells(args.cells) if args.cells else list(DEFAULT_CELLS)
    log.info("entries: %s", [_run_dirname(c, s) for c, s in entries])

    # Worker branch — in-process eval of the assigned entries.
    if args.worker_cells is not None:
        worker_entries = _parse_cells(args.worker_cells)
        log.info("worker: %d entries assigned", len(worker_entries))
        return _run_worker_in_process(worker_entries=worker_entries, args=args)

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
            for cell, seed in slice_:
                print(f"    {_run_dirname(cell, seed)}")
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "verdict": "DRY_RUN_PASS",
                    "n_entries": len(entries),
                    "n_gpus": args.gpus,
                    "partition_sizes": [len(p) for p in partitions],
                    "out_root": str(args.out_root),
                    "panel_json": str(args.panel_json),
                    "panel_json_exists": args.panel_json.exists(),
                    "questions_traj": str(args.questions_traj),
                    "questions_traj_exists": args.questions_traj.exists(),
                    "max_new_tokens": args.max_new_tokens,
                    "vllm_max_lora_rank": VLLM_MAX_LORA_RANK,
                    "env": _env_versions(),
                },
                indent=2,
            )
        )
        log.info("[phase=done] dry-run complete")
        return 0

    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.gpus == 1:
        rc = _run_worker_in_process(worker_entries=entries, args=args)
    else:
        script_path = Path(__file__).resolve()
        rc = _spawn_worker_subprocesses(partitions=partitions, args=args, script_path=script_path)

    if not args.no_aggregate:
        grid_path = _aggregate_grid(args.out_root, entries)
        log.info("grid.json → %s", grid_path)
    if rc == 0:
        # poll_pipeline.py contract: the terminal [phase=done] line marks a
        # graceful completion (a clean exit without it decays to status=dead).
        log.info("[phase=done] logit_reval complete: %d cells", len(entries))
    return rc


if __name__ == "__main__":
    sys.exit(main())
