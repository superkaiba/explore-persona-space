#!/usr/bin/env python3
"""Task #612 predictor-v3 two-phase driver (followup onpolicy-leakage-predictor).

Wraps a SINGLE pod launch and orchestrates the plan-v3 §4.2 equalize-down +
§5 KILL gate WITHOUT operator hand-orchestration (closes CONCERN
``v3-floor-n-driver-unscripted``). The round-1 dispatcher's ``--floor-n`` arg
is optional, so a bare ``--stage predictor-v3`` sweep would run
``floor_n=None`` -> variable per-source N -> the exact #601 dose confound the
plan exists to eliminate. This driver computes floor-N BEFORE any training cell
and threads it into every Phase-B cell.

Three phases, in order, with ``[phase=...]`` log markers and sentinel-file
milestones (NEVER ``scripts/task.py`` — pod-side discipline, CLAUDE.md):

  [phase=phase_a]  yield + equalize-down (BEFORE any training)
    1. per source: source-side baseline read (base agreement rate; predictor (a)
       base-prior covariate, §4.2) + on-policy pool build at REALIZED fill (no
       --floor-n). A source whose on-policy yield misses the 80% floor exits the
       pool builder with code 42 (V3YieldBelowFloor) -> recorded as a DROP,
       NEVER template-backfilled.
    2. kept = predictor_v3.kept_sources_or_kill({source: yield_decision}).
       len(kept) < V3_MIN_KEPT_SOURCES -> phase_a_kill.json sentinel +
       epm:failure(data) sentinel + clean exit (NO training launched).
    3. eq = predictor_v3.equalize_down({kept source: realized fill}); eq.floor_N
       is the cross-source training row count.
    4. persist phase_a_summary.json (per-source fill + baseline, kept, floor_N).

  [phase=phase_b]  training + eval + upload (per kept cell, --floor-n eq.floor_N)
    shells dispatch_sycophancy_612.py --stage predictor-v3 per shard. The
    dispatcher trains with the sub-epoch save cadence (round 1), runs the
    band-entry self-eval + full-panel eval at the matched-install checkpoint,
    judges, and uploads (HF data raw_completions + HF model adapter + WandB).
    CVD-pinned parallel shards (one GPU each) when --gpus lists >1 device.

  [phase=phase_c]  analysis (CPU only, post-training)
    1. issue612_predictor_bakeoff.py -> bakeoff/predictor_bakeoff.json.
    2. analyze_612.py (--stage endpoint) -> analysis_612.json, then extract the
       H1 matched-install contrast to onpolicy_predictor/h1/h1_onpolicy_vs_canned.json.

  [phase=done]  terminal line + epm:results sentinel.

Smoke == sweep with one cell: ``--cells villain:arm_onpolicy:42 --smoke`` runs
A (real CPU yield/equalize math against a fixtured or real pool_meta) -> B
(dispatcher --dry-run, no GPU) -> C (bakeoff + H1 on whatever slab exists). The
GPU-bound Phase-A pool build + baseline read decompose into the carve-out
substitute items (--skip-gpu-phase-a uses an existing pool_meta.json / baseline
record).

This driver adds NO experiment-recipe change: it only sequences the round-1
primitives (predictor_v3.kept_sources_or_kill / equalize_down, the v3 pool
builder, the dispatcher --stage predictor-v3, the bakeoff + analyze CLIs).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.sycophancy_onpolicy_612 import (  # noqa: E402
    SEEDS,
    SOURCES,
    V3_HF_DATA_PREFIX,
    V3_MIN_KEPT_SOURCES,
    V3_TRAIN_ARMS,
    V3_YIELD_FLOOR,
    cell_id,
    parse_cells,
)
from explore_persona_space.experiments.sycophancy_onpolicy_612.predictor_v3 import (  # noqa: E402
    YieldKill,
    equalize_down,
    kept_sources_or_kill,
    yield_decision,
)

log = logging.getLogger("issue612_predictor_v3_driver")

# The on-policy v3 pool builder exits with this code when the source's yield
# misses the 80% floor (build_predictor_v3_pool.V3YieldBelowFloor / the
# dispatcher's G3 drop). The driver maps it to a recorded per-source DROP.
YIELD_BELOW_FLOOR_EXIT = 42

REPO_ROOT = Path(__file__).resolve().parent.parent
DISPATCHER = REPO_ROOT / "scripts" / "dispatch_sycophancy_612.py"
BAKEOFF_CLI = REPO_ROOT / "scripts" / "issue612_predictor_bakeoff.py"
ANALYZE_MOD = "explore_persona_space.experiments.sycophancy_onpolicy_612.analyze_612"
POOL_BUILD_MOD = "explore_persona_space.experiments.sycophancy_onpolicy_612.build_predictor_v3_pool"


# ----- sentinel + git helpers (poll_pipeline-conforming) ----------------------


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except Exception:
        return "unknown"


def _write_sentinel(
    logs_root: Path,
    *,
    kind: str,
    note_obj: dict,
    name_slug: str,
    gate: str | None = None,
    version: int = 1,
) -> Path:
    """One poll_pipeline-conforming sentinel (sentinel_schema_version/kind/version),
    matching dispatch_sycophancy_612._write_sentinel byte-for-byte on the keys
    poll_pipeline._SENTINEL_REQUIRED_KEYS checks."""
    logs_root.mkdir(parents=True, exist_ok=True)
    path = logs_root / f"issue-612-{name_slug}-{int(time.time())}.json"
    payload: dict = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": version,
        "task_id": 612,
        "by": "pod-driver-612-v3",
        "ts": datetime.now(UTC).isoformat(),
        "note": json.dumps(note_obj, ensure_ascii=False),
    }
    if gate is not None:
        payload["gate"] = gate
        payload["blocks_pipeline"] = True
    path.write_text(json.dumps(payload, indent=2))
    log.info("sentinel written: %s (kind=%s)", path, kind)
    return path


def _run(
    cmd: list[str], *, env: dict[str, str] | None = None, phase_log: Path | None = None
) -> int:
    """Run a subprocess with an EXPLICIT env (uv-run / CI re-invocation safety).
    Returns the return code (does NOT raise — the caller branches on it).

    ``phase_log`` redirects the child's stdout+stderr to a per-phase log FILE
    instead of the main pod log. This is load-bearing for the inner
    dispatch_sycophancy_612 subprocess: that script emits its OWN terminal
    ``[phase=done]`` line, and ``[phase=done]`` in the MAIN pod log is RESERVED
    for THIS driver's single terminal line (poll_pipeline reads the latest
    ``[phase=...]`` token from the main log tail; a child's mid-run
    ``[phase=done]`` reaching the main log is the #545 false-done hazard).
    Phase-A pool builds / Phase-C analysis CLIs are tee'd the same way."""
    full_env = {**os.environ} if env is None else env
    log.info("spawning: %s", " ".join(cmd))
    if phase_log is None:
        proc = subprocess.run(cmd, env=full_env)
        return proc.returncode
    phase_log.parent.mkdir(parents=True, exist_ok=True)
    with open(phase_log, "ab") as fh:
        proc = subprocess.run(cmd, env=full_env, stdout=fh, stderr=subprocess.STDOUT)
    log.info("  (child stdout/stderr -> %s, rc=%d)", phase_log, proc.returncode)
    return proc.returncode


# ----- Phase A: yield + equalize-down -----------------------------------------


def _onpolicy_pool_path(data_root: Path, source: str) -> Path:
    return data_root / "onpolicy_predictor" / "training_pools" / "arm_onpolicy" / source


def _build_onpolicy_pool_for_fill(
    source: str, *, data_root: Path, gpu_id: int, judge_concurrency: int
) -> dict:
    """Build the source's on-policy pool at REALIZED fill (no --floor-n) and
    return its yield record. A below-floor yield (builder exit 42) is recorded as
    a DROP (NEVER backfilled). Idempotent: an existing pool_meta.json short-
    circuits the GPU rebuild."""
    out_dir = _onpolicy_pool_path(data_root, source)
    meta_path = out_dir / "pool_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        n_filled = int(meta["n_positives"])
        log.info(
            "[%s] [phase=phase_a] on-policy pool exists (%d pos) — idempotent skip",
            source,
            n_filled,
        )
        return {**yield_decision(source, n_filled), "tier_mix": meta.get("tier_mix", {})}

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        POOL_BUILD_MOD,
        "--source",
        source,
        "--arm",
        "arm_onpolicy",
        "--data-root",
        str(data_root),
        "--out-dir",
        str(out_dir),
        "--judge-concurrency",
        str(judge_concurrency),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id), "TQDM_DISABLE": "1"}
    log.info("[%s] [phase=phase_a] on-policy pool build (realized fill)", source)
    rc = _run(cmd, env=env, phase_log=out_dir / "pool_build.log")
    if rc == YIELD_BELOW_FLOOR_EXIT:
        # Below the 80% floor: a DROP. The builder exits before writing the pool,
        # so the exact count is not persisted; the kept/kill decision needs only
        # the drop verdict (yield_decision(n<floor) -> drop for any such n).
        log.warning(
            "[%s] [phase=phase_a] on-policy yield BELOW 80%% floor (exit 42) — DROP", source
        )
        rec = yield_decision(source, V3_YIELD_FLOOR - 1)
        rec["below_floor_drop"] = True
        return rec
    if rc != 0:
        raise RuntimeError(
            f"[{source}] on-policy pool build failed rc={rc} (not the yield-drop code)"
        )
    if not meta_path.exists():
        raise RuntimeError(f"[{source}] pool build exited 0 but {meta_path} missing")
    meta = json.loads(meta_path.read_text())
    n_filled = int(meta["n_positives"])
    log.info("[%s] [phase=phase_a] realized fill = %d positives", source, n_filled)
    return {**yield_decision(source, n_filled), "tier_mix": meta.get("tier_mix", {})}


def run_phase_a(args: argparse.Namespace, sources: list[str]) -> dict | None:
    """Phase A: per-source baseline read + on-policy yield, then kept-or-kill +
    equalize-down. Returns the phase_a_summary dict on KEEP, or ``None`` after
    writing the KILL sentinel (caller exits cleanly without training)."""
    log.info("[phase=phase_a] yield + equalize-down for sources=%s", sources)
    slab_root: Path = args.slab_root
    data_root: Path = args.data_root

    # 1a. Source-side baseline reads (GPU eval; predictor (a) base-prior covariate).
    baselines: dict[str, dict] = {}
    if not args.skip_gpu_phase_a:
        # Imported lazily so the CPU smoke (--skip-gpu-phase-a) never constructs
        # the GPU runner / its heavy deps.
        from dispatch_sycophancy_612 import PredictorV3Runner

        runner = PredictorV3Runner(args)
        for source in sources:
            runner.run_source_baseline(source)

    base_dir = slab_root / "onpolicy_predictor" / "source_baseline"
    for source in sources:
        rec_path = base_dir / f"{source}.json"
        if rec_path.exists():
            baselines[source] = json.loads(rec_path.read_text())
        else:
            baselines[source] = {"base_agreement_rate": None, "note": "baseline record absent"}

    # 1b. On-policy yield per source (realized fill / drop).
    decisions: dict[str, dict] = {}
    if args.skip_gpu_phase_a:
        # CPU carve-out: read realized fill from existing pool_meta.json sidecars
        # (the GPU pool build ran elsewhere / in a prior invocation).
        for source in sources:
            meta_path = _onpolicy_pool_path(data_root, source) / "pool_meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"--skip-gpu-phase-a but {meta_path} absent — the on-policy pool build must "
                    f"have run (GPU) before the CPU equalize-down can read realized fill"
                )
            meta = json.loads(meta_path.read_text())
            decisions[source] = {
                **yield_decision(source, int(meta["n_positives"])),
                "tier_mix": meta.get("tier_mix", {}),
            }
    else:
        for source in sources:
            decisions[source] = _build_onpolicy_pool_for_fill(
                source,
                data_root=data_root,
                gpu_id=args.gpu_id,
                judge_concurrency=args.judge_concurrency,
            )

    # 2. Kept-or-kill (§5).
    out_dir = slab_root / "onpolicy_predictor"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        kept = kept_sources_or_kill(decisions)
    except YieldKill as kill:
        dropped = sorted(s for s, d in decisions.items() if d["decision"] == "drop")
        kept_sources = sorted(s for s, d in decisions.items() if d["decision"] == "keep")
        kill_note = {
            "event": "phase_a_kill",
            "reason": "yield_below_floor_kill_gate_triggered",
            "kept": kept_sources,
            "dropped": dropped,
            "min_kept_required": V3_MIN_KEPT_SOURCES,
            "decisions": decisions,
            "message": str(kill),
            "git_commit_sha": _git_sha(),
        }
        (out_dir / "phase_a_kill.json").write_text(json.dumps(kill_note, indent=2))
        _write_sentinel(
            args.logs_root,
            kind="epm:failure",
            name_slug="phase-a-kill",
            note_obj={"failure_class": "data", **kill_note},
        )
        log.error("[phase=phase_a] KILL: %s", kill)
        return None

    # 3. Equalize-down across kept sources.
    kept_fills = {s: decisions[s]["n_filled"] for s in kept}
    eq = equalize_down(kept_fills)

    # 4. Persist phase_a_summary.json.
    summary = {
        "schema_version": 1,
        "followup_label": "onpolicy-leakage-predictor",
        "kept_sources": kept,
        "dropped_sources": sorted(s for s, d in decisions.items() if d["decision"] == "drop"),
        "floor_n_positives": eq["floor_n_positives"],
        "n_negatives_total": eq["n_negatives_total"],
        "ratio_pos_to_neg": eq["ratio_pos_to_neg"],
        "equalize_down": eq,
        "per_source": {
            s: {
                "yield": decisions[s],
                "base_agreement_rate": baselines.get(s, {}).get("base_agreement_rate"),
            }
            for s in sources
        },
        "git_commit_sha": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    summary_path = out_dir / "phase_a_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(
        "[phase=phase_a] KEEP %s | floor_N=%d (negs=%d, ratio %.3f) -> %s",
        kept,
        eq["floor_n_positives"],
        eq["n_negatives_total"],
        eq["ratio_pos_to_neg"],
        summary_path,
    )

    if not args.skip_gpu_phase_a and os.environ.get("HF_TOKEN"):
        from explore_persona_space.orchestrate.hub import _upload  # type: ignore

        _upload(
            summary_path,
            repo_id="superkaiba1/explore-persona-space-data",
            repo_type="dataset",
            path_in_repo=f"{V3_HF_DATA_PREFIX}/phase_a_summary.json",
        )
    return summary


# ----- Phase B: training + eval + upload --------------------------------------


def _kept_cells(kept_sources: list[str]) -> list[tuple[str, str, int]]:
    """Phase-B train grid restricted to kept sources: kept x {canned,onpolicy} x
    {42,137}."""
    return [
        (source, arm, seed) for source in kept_sources for arm in V3_TRAIN_ARMS for seed in SEEDS
    ]


def _dispatcher_cmd(
    args: argparse.Namespace,
    *,
    cells: list[tuple[str, str, int]],
    all_cells: list[tuple[str, str, int]],
    floor_n: int,
    gpu_id: int,
    finalize: bool,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        str(DISPATCHER),
        "--stage",
        "predictor-v3",
        "--cells",
        ",".join(cell_id(*c) for c in cells),
        "--all-cells",
        ",".join(cell_id(*c) for c in all_cells),
        "--floor-n",
        str(floor_n),
        "--gpu-id",
        str(gpu_id),
        "--data-root",
        str(args.data_root),
        "--adapters-root",
        str(args.adapters_root),
        "--slab-root",
        str(args.slab_root),
        "--runs-root",
        str(args.runs_root),
        "--logs-root",
        str(args.logs_root),
        "--judge-concurrency",
        str(args.judge_concurrency),
        "--finalize" if finalize else "--no-finalize",
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.skip_prefetch:
        cmd.append("--skip-prefetch")
    if not args.smoke_gates:
        cmd.append("--no-smoke-gates")
    if not args.hf_upload:
        cmd.append("--no-hf-upload")
    return cmd


def run_phase_b(
    args: argparse.Namespace, kept_sources: list[str], floor_n: int
) -> list[tuple[str, str, int]]:
    """Phase B: train + eval + upload every kept cell at the equalized floor-N.

    Shells the round-1 dispatcher --stage predictor-v3. When --gpus lists >1
    device, CVD-pinned parallel shards (one GPU each, matching round-1's shard
    pattern) — each shard pins CUDA_VISIBLE_DEVICES in its env AND passes the
    matching --gpu-id so the in-process clobber in train/sft.py cannot co-locate
    cells on GPU 0. The LAST shard to finish carries --finalize; the rest pass
    --no-finalize."""
    all_cells = _kept_cells(kept_sources)
    cells = [c for c in args.cells if c in all_cells] if args.cells else all_cells
    if not cells:
        raise ValueError(
            f"no requested cells fall inside the kept-source grid {kept_sources} "
            f"(--cells={[cell_id(*c) for c in (args.cells or [])]})"
        )
    log.info(
        "[phase=phase_b] train grid: cells=%s all_cells=%d floor_n=%d gpus=%s",
        [cell_id(*c) for c in cells],
        len(all_cells),
        floor_n,
        args.gpus,
    )

    gpus = args.gpus
    if len(gpus) <= 1:
        gpu_id = gpus[0] if gpus else args.gpu_id
        rc = _run(
            _dispatcher_cmd(
                args,
                cells=cells,
                all_cells=all_cells,
                floor_n=floor_n,
                gpu_id=gpu_id,
                finalize=True,
            ),
            phase_log=args.logs_root / "phase_b_dispatch.log",
        )
        if rc != 0:
            raise RuntimeError(f"[phase=phase_b] dispatcher failed rc={rc}")
        return all_cells

    # Parallel CVD-pinned shards: round-robin cells across GPUs, run concurrently.
    shards: dict[int, list[tuple[str, str, int]]] = {g: [] for g in gpus}
    for i, c in enumerate(cells):
        shards[gpus[i % len(gpus)]].append(c)
    shards = {g: cs for g, cs in shards.items() if cs}
    last_gpu = list(shards)[-1]
    procs: list[tuple[int, subprocess.Popen, object]] = []
    for gpu_id, shard_cells in shards.items():
        cmd = _dispatcher_cmd(
            args,
            cells=shard_cells,
            all_cells=all_cells,
            floor_n=floor_n,
            gpu_id=gpu_id,
            finalize=(gpu_id == last_gpu),
        )
        # CVD pin in the LAUNCHER env per shard (one GPU each) + matching --gpu-id.
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        # Per-shard log FILE: the inner dispatcher's terminal [phase=done] must NOT
        # reach the main pod log (reserved for the driver's terminal line, #545).
        shard_log = args.logs_root / f"phase_b_dispatch_gpu{gpu_id}.log"
        shard_log.parent.mkdir(parents=True, exist_ok=True)
        log.info(
            "[phase=phase_b] shard gpu=%d cells=%s", gpu_id, [cell_id(*c) for c in shard_cells]
        )
        log.info("spawning: %s  (-> %s)", " ".join(cmd), shard_log)

        # (closed in the wait() loop below after the matching Popen exits).
        fh = open(shard_log, "ab")  # noqa: SIM115
        procs.append(
            (gpu_id, subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT), fh)
        )

    failures: list[tuple[int, int]] = []
    for gpu_id, proc, fh in procs:
        rc = proc.wait()
        fh.close()
        if rc != 0:
            failures.append((gpu_id, rc))
    if failures:
        raise RuntimeError(f"[phase=phase_b] shard(s) failed: {failures}")
    return all_cells


# ----- Phase C: analysis (CPU only) -------------------------------------------


def run_phase_c(args: argparse.Namespace) -> dict:
    """Phase C: bake-off + H1 matched-install contrast over the kept cells (CPU).

    Reuses issue612_predictor_bakeoff.py (the 3-predictor matrix) and
    analyze_612.py (--stage endpoint, which computes h1_onpolicy_vs_canned). The
    H1 sub-dict is extracted to onpolicy_predictor/h1/h1_onpolicy_vs_canned.json
    for the clean-result re-fold."""
    slab_root: Path = args.slab_root
    log.info("[phase=phase_c] bake-off + H1 matched-install contrast")

    bakeoff_out = slab_root / "onpolicy_predictor" / "bakeoff" / "predictor_bakeoff.json"
    rc = _run(
        [
            "uv",
            "run",
            "python",
            str(BAKEOFF_CLI),
            "--slab-root",
            str(slab_root),
            "--out",
            str(bakeoff_out),
            "--judge-concurrency",
            str(args.judge_concurrency),
        ],
        phase_log=args.logs_root / "phase_c_bakeoff.log",
    )
    if rc != 0:
        raise RuntimeError(f"[phase=phase_c] bake-off failed rc={rc}")

    analyze_out = slab_root / "analysis_612.json"
    rc = _run(
        [
            "uv",
            "run",
            "python",
            "-m",
            ANALYZE_MOD,
            "--slab-root",
            str(slab_root),
            "--stage",
            "endpoint",
            "--skip-figures",
        ],
        phase_log=args.logs_root / "phase_c_analyze.log",
    )
    # analyze_612 returns 2 on the K1 parity HARD_FAIL kill (analysis_612.json is
    # written first for evidence). That is a rig-validity kill, not a driver bug —
    # re-raise so the orchestrator's failure routing sees it.
    if rc not in (0, 2):
        raise RuntimeError(f"[phase=phase_c] analyze_612 failed rc={rc}")
    if not analyze_out.exists():
        raise RuntimeError(f"[phase=phase_c] analyze_612 exited {rc} but {analyze_out} missing")

    analysis = json.loads(analyze_out.read_text())
    h1 = {
        "schema_version": 1,
        "followup_label": "onpolicy-leakage-predictor",
        "h1_onpolicy_vs_canned": analysis.get("h1_onpolicy_vs_canned"),
        "h1_onpolicy_vs_canned_per_source": analysis.get("h1_onpolicy_vs_canned_per_source"),
        "missing_cells": analysis.get("missing_cells"),
        "source_analysis": str(analyze_out.relative_to(REPO_ROOT))
        if analyze_out.is_relative_to(REPO_ROOT)
        else str(analyze_out),
        "git_commit_sha": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    h1_out = slab_root / "onpolicy_predictor" / "h1" / "h1_onpolicy_vs_canned.json"
    h1_out.parent.mkdir(parents=True, exist_ok=True)
    h1_out.write_text(json.dumps(h1, indent=2))
    log.info(
        "[phase=phase_c] bakeoff -> %s | H1 verdict=%s -> %s",
        bakeoff_out,
        (h1["h1_onpolicy_vs_canned"] or {}).get("verdict"),
        h1_out,
    )
    return {
        "bakeoff_json": str(bakeoff_out),
        "h1_json": str(h1_out),
        "analysis_json": str(analyze_out),
        "parity_hard_fail": rc == 2,
    }


# ----- orchestration ----------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task #612 predictor-v3 two-phase driver (floor-N equalize-down + KILL gate).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sources",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=list(SOURCES),
        help=f"Phase-A source personas (default all 4: {','.join(SOURCES)}).",
    )
    parser.add_argument(
        "--cells",
        type=parse_cells,
        default=None,
        help="Phase-B cell subset <source>:<arm>:<seed> (default: every kept cell). "
        "Smoke: villain:arm_onpolicy:42.",
    )
    parser.add_argument(
        "--gpus",
        type=lambda s: [int(x) for x in s.split(",") if x.strip() != ""],
        default=[],
        help="Phase-B parallel-shard GPU ids (CVD-pinned, one GPU each). Empty -> "
        "single shard on --gpu-id.",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="Single-shard / Phase-A GPU id.")
    parser.add_argument("--data-root", type=Path, default=Path("data/issue_612"))
    parser.add_argument("--adapters-root", type=Path, default=Path("/workspace/adapters_411"))
    parser.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_612"))
    parser.add_argument("--runs-root", type=Path, default=Path("/workspace/runs/issue_612"))
    parser.add_argument("--logs-root", type=Path, default=Path("/workspace/logs"))
    parser.add_argument("--judge-concurrency", type=int, default=16)
    parser.add_argument(
        "--floor-n-override",
        type=int,
        default=None,
        help="Skip Phase-A yield/equalize and force floor-N (smoke / re-run after a "
        "known Phase-A result). Phase A still runs the baseline reads unless "
        "--skip-phase-a.",
    )
    parser.add_argument(
        "--skip-phase-a",
        action="store_true",
        help="Skip Phase A entirely (re-run Phase B/C from a persisted "
        "phase_a_summary.json). Requires --floor-n-override or an existing summary.",
    )
    parser.add_argument(
        "--skip-gpu-phase-a",
        action="store_true",
        help="Carve-out: run Phase-A yield/equalize math on EXISTING pool_meta.json + "
        "baseline records (no GPU pool build / baseline eval). CPU smoke path.",
    )
    parser.add_argument("--skip-phase-b", action="store_true", help="Skip training (Phase B).")
    parser.add_argument("--skip-phase-c", action="store_true", help="Skip analysis (Phase C).")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Convenience: --skip-gpu-phase-a + Phase-B --dry-run + single-cell. Real "
        "CPU phases (A math, C) run; GPU phases are dry-run / skipped.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Phase-B dispatcher --dry-run.")
    parser.add_argument("--skip-prefetch", action="store_true")
    parser.add_argument("--smoke-gates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hf-upload", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    if args.smoke:
        args.skip_gpu_phase_a = True
        args.dry_run = True

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    os.environ.setdefault("WANDB_PROJECT", "issue612_sycophancy_onpolicy")
    os.environ.setdefault("TQDM_DISABLE", "1")

    log.info(
        "[phase=dispatch] predictor-v3 driver sources=%s gpus=%s dry_run=%s skip_gpu_phase_a=%s",
        args.sources,
        args.gpus,
        args.dry_run,
        args.skip_gpu_phase_a,
    )

    # ----- Phase A -----
    summary: dict | None = None
    summary_path = args.slab_root / "onpolicy_predictor" / "phase_a_summary.json"
    if args.skip_phase_a:
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            log.info("[phase=phase_a] SKIPPED — reusing %s", summary_path)
        elif args.floor_n_override is None:
            raise ValueError(
                "--skip-phase-a requires an existing phase_a_summary.json or --floor-n-override"
            )
    else:
        summary = run_phase_a(args, args.sources)
        if summary is None:
            # KILL gate fired; phase_a_kill.json + epm:failure sentinel written.
            log.info("[phase=done]")  # clean terminal: NO training launched
            return 0

    floor_n = args.floor_n_override
    kept_sources = list(args.sources)
    if summary is not None:
        kept_sources = summary["kept_sources"]
        if floor_n is None:
            floor_n = summary["floor_n_positives"]
    if floor_n is None:
        raise ValueError(
            "floor-N unresolved — Phase A produced no summary and no --floor-n-override"
        )

    # ----- Phase B -----
    if not args.skip_phase_b:
        run_phase_b(args, kept_sources, floor_n)
    else:
        log.info("[phase=phase_b] SKIPPED")

    # ----- Phase C -----
    phase_c: dict = {}
    if not args.skip_phase_c:
        phase_c = run_phase_c(args)
    else:
        log.info("[phase=phase_c] SKIPPED")

    # ----- finalize: epm:results sentinel + terminal line -----
    results_note = {
        "event": "predictor_v3_driver_complete",
        "followup_label": "onpolicy-leakage-predictor",
        "kept_sources": kept_sources,
        "floor_n_positives": floor_n,
        "phase_a_summary": str(summary_path) if summary is not None else None,
        "phase_c": phase_c,
        "git_commit_sha": _git_sha(),
    }
    _write_sentinel(
        args.logs_root, kind="epm:results", name_slug="epm_results", note_obj=results_note
    )
    log.info("[phase=done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
