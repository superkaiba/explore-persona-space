#!/usr/bin/env python3
"""End-to-end pipeline dispatcher for #519.

PASS_UNIFIED architectural-parity: smoke = sweep with `--cells N
--seeds N` parameterization. The same dispatcher orchestrates the full
pipeline in either mode; the smoke path is just the sweep with `cells
= 1` (one EM canary cell) and `seeds = 1` (seed=42 only). Step 6d.0
gate per `.claude/agents/experiment-implementer.md`.

Phases (each is checkpointed; phase output written before next phase
starts — `checkpoint per phase` rule):

    A1. data-gen Step Z (vLLM aligned-negative regen) — once, cached
    A2. data-gen marker arm (3 seeds, vLLM base-response generation)
    A3. data-gen em arm (3 seeds, build contrastive JSONL)
    B1. Phase-0 EM smoke (1 cell, max_steps=50, saturation check)
    B2. marker arm training (3 seeds, 1 GPU each, parallel within wave)
    B3. em arm training (3 seeds, 1 GPU each, parallel within wave)
    C.  Activation-shift extraction (6 cells x 3 variants)
    D.  SVD direction-constancy + cosine / norm-regression analyses
    E.  Steering-vector extraction (CAA, disjoint pools, 1 per arm)
    F.  Aggregate JSON written to ``eval_results/issue_519/``

Each cell is launched via ``subprocess.Popen`` with the right
``CUDA_VISIBLE_DEVICES`` (the `feedback_cvd_hydra_override` rule). The
training step uses ``--gpu-id N``; the activation-shift / steering
steps use the env var directly.

CLI:
    # Full sweep:
    nohup uv run python scripts/issue_519_dispatch.py \
        --mode sweep \
        --output-dir /workspace/eval_results/issue_519 \
        > /workspace/logs/issue-519-train.log 2>&1 &

    # Smoke (single-cell canary; UNIFIED with sweep, just narrower):
    uv run python scripts/issue_519_dispatch.py --mode smoke

The smoke pathway is the SAME code path as the sweep; the only
differences are ``--cells 1 --seeds 1 --max-train-steps 2 --cpu-only``
flag overrides.  No subprocess-vs-in-process divergence (Step 6d.0
verdict = PASS_UNIFIED).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ARMS: tuple[str, ...] = ("marker", "em")
DEFAULT_SEEDS: tuple[int, ...] = (42, 137, 256)
DEFAULT_VARIANTS: tuple[str, ...] = ("same", "base", "on_policy")


def _resolve_repo_root() -> Path:
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
    return Path(out)


@dataclass(frozen=True)
class Cell:
    arm: str
    seed: int
    gpu_id: int


def _build_cells(arms: Sequence[str], seeds: Sequence[int], n_gpus: int) -> list[Cell]:
    """Round-robin assignment of (arm, seed) cells to GPU ids.

    Same shape as the plan §4.2 dispatcher pseudocode; we override only
    the cell count + seed count for smoke (smoke = the same dispatcher,
    just fewer cells).
    """
    cells: list[Cell] = []
    pos = 0
    for arm in arms:
        for seed in seeds:
            cells.append(Cell(arm=arm, seed=seed, gpu_id=pos % max(n_gpus, 1)))
            pos += 1
    return cells


def _run_with_log(
    cmd: Sequence[str],
    *,
    log_path: Path,
    extra_env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> int:
    """Run a child process, tee stdout/stderr to a log file. Returns rc."""
    env = {**os.environ}
    if extra_env:
        env.update(extra_env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "$ %s  >>> %s%s",
        " ".join(shlex.quote(c) for c in cmd),
        log_path,
        f" (env+={list(extra_env.keys())})" if extra_env else "",
    )
    with log_path.open("ab") as f:
        proc = subprocess.run(
            list(cmd),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
    rc = proc.returncode
    if rc != 0:
        logger.error("command exited with rc=%d (log: %s)", rc, log_path)
    return rc


def _run_parallel_with_log(
    cmds: Iterable[tuple[Sequence[str], Path, dict[str, str] | None]],
    *,
    cwd: Path | None = None,
) -> list[int]:
    """Run several subprocesses concurrently. Returns parallel list of rc codes.

    Each entry is (cmd_argv, log_path, extra_env).
    """
    procs: list[subprocess.Popen] = []
    files = []
    for cmd, log_path, extra_env in cmds:
        env = {**os.environ}
        if extra_env:
            env.update(extra_env)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f = log_path.open("ab")
        files.append(f)
        logger.info(
            "$ (parallel) %s  >>> %s%s",
            " ".join(shlex.quote(c) for c in cmd),
            log_path,
            f" (env+={list(extra_env.keys())})" if extra_env else "",
        )
        p = subprocess.Popen(
            list(cmd),
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
        procs.append(p)
    rcs = [p.wait() for p in procs]
    for f in files:
        f.close()
    return rcs


def phase_a1_step_z(
    *,
    repo_root: Path,
    n_positives: int,
    out_path: Path,
    log_dir: Path,
    dry_run: bool,
) -> None:
    """Phase A1: Step Z aligned-negative regen (vLLM, 1xGPU)."""
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue_519_em_aligned_neg_regen.py",
        "--n-positives",
        str(n_positives),
        "--shuffle-seed",
        "0",
        "--out",
        str(out_path),
    ]
    if dry_run:
        cmd.append("--dry-run")
    log_path = log_dir / "phase_a1_step_z.log"
    rc = _run_with_log(cmd, log_path=log_path, cwd=repo_root)
    if rc != 0:
        raise RuntimeError(f"phase A1 (Step Z) failed with rc={rc}; see {log_path}")
    logger.info("[phase=a1_done]")


def phase_a2_a3_build_data(
    *,
    repo_root: Path,
    arms: Sequence[str],
    seeds: Sequence[int],
    n_positives: int,
    n_negs_per_persona: int,
    log_dir: Path,
    em_aligned_negs: Path,
    smoke_fake_responses: bool,
    allow_stub_responses: bool,
    marker_question_pool: Path | None = None,
) -> None:
    """Phase A2/A3: build contrastive training JSONLs for each (arm, seed)."""
    for arm in arms:
        for seed in seeds:
            out = repo_root / "data" / "issue_519" / f"{arm}_seed{seed}.jsonl"
            cmd = [
                "uv",
                "run",
                "python",
                "scripts/issue_519_build_data.py",
                "--arm",
                arm,
                "--seed",
                str(seed),
                "--out",
                str(out),
                "--n-positives",
                str(n_positives),
                "--n-negatives-per-persona",
                str(n_negs_per_persona),
            ]
            if arm == "em":
                cmd.extend(["--em-aligned-negs", str(em_aligned_negs)])
                if allow_stub_responses:
                    cmd.append("--allow-stub-responses")
            if arm == "marker":
                if marker_question_pool is not None:
                    cmd.extend(["--question-pool", str(marker_question_pool)])
                if smoke_fake_responses:
                    cmd.append("--smoke-fake-responses")
            log_path = log_dir / f"phase_a23_build_{arm}_seed{seed}.log"
            rc = _run_with_log(cmd, log_path=log_path, cwd=repo_root)
            if rc != 0:
                raise RuntimeError(f"phase A2/A3 ({arm} seed={seed}) failed; see {log_path}")
    logger.info("[phase=a23_done]")


def phase_b0_saturation_gate(
    *,
    repo_root: Path,
    smoke_output_dir: Path,
    seed: int,
    arm: str,
    log_dir: Path,
    saturation_cfg: dict,
) -> None:
    """Phase B0 post-train DV eval + saturation gate (round-1 reviewer M1 fix).

    Reads `smoke_output_dir/{arm}_seed{seed}/adapter` (the 50-step
    smoke checkpoint), runs the arm's DV eval on the held-out 20-question
    pool, parses the result against the plan §4.3 thresholds:

    - Marker arm: `lower_nats <= log_p_marker_delta_source <= upper_nats`
      (plan §4.3 marker arm — out-of-band -> retry on the lr ladder
      [5e-7, 1e-6, 2e-6]).
    - EM arm: `lower_pct <= aligned_coherent_rate_source <= upper_pct`
      (plan §4.3 EM arm — out-of-band -> plan revision required).

    On OUT-OF-BAND result, raises with a clear message naming the
    retry path the plan specifies; the dispatcher exits non-zero and
    the orchestrator/experimenter handles the retry (per plan §4.3 the
    retry is a fresh /issue invocation, NOT an auto-retry inside the
    dispatcher).
    """
    import yaml as _yaml

    cell_dir = smoke_output_dir / f"{arm}_seed{seed}"
    adapter_dir = cell_dir / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Phase B0 smoke adapter not found at {adapter_dir} — "
            f"training cell did not write the adapter (check {log_dir}/phase_b_train_*.log)."
        )

    log_path = log_dir / f"phase_b0_saturation_gate_{arm}_seed{seed}.log"
    result_json = cell_dir / "saturation_gate_result.json"

    if arm == "marker":
        # Load the condition YAML to get marker text.
        cond_yaml = repo_root / "configs" / "condition" / f"c_issue_519_{arm}.yaml"
        with cond_yaml.open() as f:
            cond_cfg = _yaml.safe_load(f)
        marker_text = cond_cfg.get("marker_token", " ※")
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/issue_519_marker_gate_eval.py",
            "--adapter-dir",
            str(adapter_dir),
            "--persona",
            "medical_doctor",
            "--out",
            str(result_json),
            "--marker-text",
            marker_text,
        ]
        rc = _run_with_log(cmd, log_path=log_path, cwd=repo_root)
        if rc != 0:
            raise RuntimeError(f"phase B0 saturation-gate eval failed; see {log_path}")
        with result_json.open() as f:
            r = json.load(f)
        delta = float(r["log_p_marker_delta_source"])
        lo = float(saturation_cfg.get("lower_nats", 5.0))
        hi = float(saturation_cfg.get("upper_nats", 12.0))
        ladder = saturation_cfg.get("lr_ladder", [5.0e-7, 1.0e-6, 2.0e-6])
        logger.info(
            "[phase=b0_gate arm=marker] log_p_marker_delta_source=%.3f nats (band=[%.1f,%.1f])",
            delta,
            lo,
            hi,
        )
        if delta < lo:
            raise RuntimeError(
                f"Phase B0 saturation gate FAIL (marker, seed={seed}): "
                f"log_p_marker_delta_source={delta:.3f} nats < lower_bound={lo} "
                f"-- training under-shot. Per plan §4.3, retry on the lr ladder "
                f"{ladder} (up-step), then re-invoke /issue."
            )
        if delta > hi:
            raise RuntimeError(
                f"Phase B0 saturation gate FAIL (marker, seed={seed}): "
                f"log_p_marker_delta_source={delta:.3f} nats > upper_bound={hi} "
                f"-- training saturated. Per plan §4.3, retry on the lr ladder "
                f"{ladder} (down-step), then re-invoke /issue."
            )
        logger.info("[phase=b0_gate arm=marker] PASS (in-band)")
        return

    # EM arm: run the Sonnet Betley judge on the smoke adapter's
    # source-persona generations.
    cond_yaml = repo_root / "configs" / "condition" / f"c_issue_519_{arm}.yaml"
    with cond_yaml.open() as f:
        cond_cfg = _yaml.safe_load(f)
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue_519_em_gate_eval.py",
        "--adapter-dir",
        str(adapter_dir),
        "--persona",
        "medical_doctor",
        "--out",
        str(result_json),
    ]
    rc = _run_with_log(cmd, log_path=log_path, cwd=repo_root)
    if rc != 0:
        raise RuntimeError(f"phase B0 EM saturation-gate eval failed; see {log_path}")
    with result_json.open() as f:
        r = json.load(f)
    rate = float(r["aligned_coherent_rate_source"])
    lo = float(saturation_cfg.get("lower_pct", 3.0))
    hi = float(saturation_cfg.get("upper_pct", 25.0))
    logger.info(
        "[phase=b0_gate arm=em] aligned_coherent_rate_source=%.1f%% (band=[%.1f,%.1f]%%)",
        rate * 100,
        lo,
        hi,
    )
    if rate * 100 < lo:
        raise RuntimeError(
            f"Phase B0 saturation gate FAIL (em, seed={seed}): "
            f"aligned_coherent_rate_source={rate * 100:.1f}% < lower_bound={lo}% "
            f"-- training under-shot Betley AND/coherent. Per plan §4.3 this requires "
            f"a plan revision (lr ladder doesn't help -- the EM positives need investigation)."
        )
    if rate * 100 > hi:
        raise RuntimeError(
            f"Phase B0 saturation gate FAIL (em, seed={seed}): "
            f"aligned_coherent_rate_source={rate * 100:.1f}% > upper_bound={hi}% "
            f"-- training saturated. Per plan §4.3 retry on the lr down-step ladder."
        )
    logger.info("[phase=b0_gate arm=em] PASS (in-band)")


def phase_b_train(
    *,
    repo_root: Path,
    cells: Sequence[Cell],
    output_dir: Path,
    log_dir: Path,
    max_steps_override: int | None,
    skip_callbacks: bool,
    no_hf_upload: bool,
    cpu_only: bool,
    n_gpus: int,
) -> None:
    """Phase B: 6 LoRA SFT cells, n_gpus in parallel per wave."""
    for wave_start in range(0, len(cells), max(n_gpus, 1)):
        wave = cells[wave_start : wave_start + max(n_gpus, 1)]
        commands: list[tuple[Sequence[str], Path, dict[str, str] | None]] = []
        for cell in wave:
            data_path = repo_root / "data" / "issue_519" / f"{cell.arm}_seed{cell.seed}.jsonl"
            out_dir = output_dir / f"{cell.arm}_seed{cell.seed}"
            cmd = [
                "uv",
                "run",
                "python",
                "scripts/issue_519_train.py",
                "--arm",
                cell.arm,
                "--seed",
                str(cell.seed),
                "--data-path",
                str(data_path),
                "--output-dir",
                str(out_dir),
                "--gpu-id",
                str(cell.gpu_id),
            ]
            if max_steps_override is not None:
                cmd.extend(["--max-steps", str(max_steps_override)])
            if skip_callbacks:
                cmd.append("--skip-callbacks")
            if no_hf_upload:
                cmd.append("--no-hf-upload")
            if cpu_only:
                cmd.append("--cpu-only")
            log_path = log_dir / f"phase_b_train_{cell.arm}_seed{cell.seed}.log"
            commands.append((cmd, log_path, None))
        rcs = _run_parallel_with_log(commands, cwd=repo_root)
        bad = [(rc, c) for rc, c in zip(rcs, wave, strict=True) if rc != 0]
        if bad:
            raise RuntimeError(f"phase B wave training failed: {bad}; see logs in {log_dir}")
    logger.info("[phase=b_done]")


def phase_c_extract_shifts(
    *,
    repo_root: Path,
    cells: Sequence[Cell],
    variants: Sequence[str],
    output_dir: Path,
    log_dir: Path,
    layer: int,
    personas_json: Path,
    questions_json: Path,
    n_gpus: int,
    cpu_only: bool,
    adapter_dir_override: dict[Cell, Path] | None = None,
) -> None:
    """Phase C: activation-shift extraction per (arm, seed, variant)."""
    for variant in variants:
        for wave_start in range(0, len(cells), max(n_gpus, 1)):
            wave = cells[wave_start : wave_start + max(n_gpus, 1)]
            commands: list[tuple[Sequence[str], Path, dict[str, str] | None]] = []
            for cell in wave:
                # Adapter path: trainer wrote it to output_dir/{arm}_seed{S}/adapter
                if adapter_dir_override and cell in adapter_dir_override:
                    adapter_path = adapter_dir_override[cell]
                else:
                    adapter_path = output_dir / f"{cell.arm}_seed{cell.seed}" / "adapter"
                shift_out = output_dir / "shifts" / f"{variant}_{cell.arm}_seed{cell.seed}.pt"
                cmd = [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "explore_persona_space.analysis.activation_shift",
                    "--arm",
                    cell.arm,
                    "--seed",
                    str(cell.seed),
                    "--variant",
                    variant,
                    "--layer",
                    str(layer),
                    "--adapter-path",
                    str(adapter_path),
                    "--personas-json",
                    str(personas_json),
                    "--questions-json",
                    str(questions_json),
                    "--out",
                    str(shift_out),
                ]
                env = {"CUDA_VISIBLE_DEVICES": "" if cpu_only else str(cell.gpu_id)}
                log_path = log_dir / f"phase_c_{variant}_{cell.arm}_seed{cell.seed}.log"
                commands.append((cmd, log_path, env))
            rcs = _run_parallel_with_log(commands, cwd=repo_root)
            bad = [(rc, c) for rc, c in zip(rcs, wave, strict=True) if rc != 0]
            if bad:
                raise RuntimeError(
                    f"phase C extraction failed at variant={variant}: {bad}; see logs in {log_dir}"
                )
    logger.info("[phase=c_done]")


def phase_d_svd_analysis(
    *,
    repo_root: Path,
    cells: Sequence[Cell],
    variants: Sequence[str],
    output_dir: Path,
    log_dir: Path,
    base_cosines_json: Path | None,
) -> None:
    """Phase D: in-process SVD + null + cosine + regression analyses.

    Reads the per-cell shift .pt files written in Phase C, assembles M
    per (arm, seed, variant), runs ``svd_summary`` + nulls, writes a
    per-cell JSON + one aggregate.
    """
    import numpy as np
    import torch

    from explore_persona_space.analysis.svd_direction_constancy import (
        assemble_M,
        bootstrap_ci,
        row_shuffle_null,
        shift_norm_vs_cosine_regression,
        sign_flip_null,
        svd_summary,
    )

    out_dir = output_dir / "svd"
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate: dict[str, dict] = {}
    for variant in variants:
        for cell in cells:
            shift_path = output_dir / "shifts" / f"{variant}_{cell.arm}_seed{cell.seed}.pt"
            if not shift_path.exists():
                logger.warning("missing shift file %s — skipping", shift_path)
                continue
            payload = torch.load(shift_path, map_location="cpu", weights_only=False)
            shifts = payload["shifts"]
            M, persona_order = assemble_M(shifts)
            svd = svd_summary(M)
            row_null = row_shuffle_null(M, n_reps=1000, seed=cell.seed)
            sign_null = sign_flip_null(M, n_reps=1000, seed=cell.seed)
            entry: dict = {
                "variant": variant,
                "arm": cell.arm,
                "seed": cell.seed,
                "M_shape": list(svd["M_shape"]),
                "persona_order": persona_order,
                "s_top1_frac": svd["s_top1_frac"],
                "row_shuffle_p95": row_null["p95"],
                "row_shuffle_p99": row_null["p99"],
                "sign_flip_p95": sign_null["p95"],
                "sign_flip_p99": sign_null["p99"],
                "mean_cos_to_U1": float(np.mean(svd["cos_to_U1"])),
                "median_cos_to_U1": float(np.median(svd["cos_to_U1"])),
                "cos_to_U1": svd["cos_to_U1"].tolist(),
                "singular_values": svd["s"].tolist(),
            }
            if base_cosines_json is not None and base_cosines_json.exists():
                with base_cosines_json.open() as f:
                    base_cos = json.load(f)
                # Round-1 reviewer M3 fix: refuse to silently default missing
                # personas to 0.0 — a stale or incomplete base-cosines artifact
                # would fabricate Spearman inputs as zeros.
                missing = [p for p in persona_order if p not in base_cos]
                if missing:
                    raise KeyError(
                        f"base-cosines JSON {base_cosines_json} is missing entries for "
                        f"persona(s) {missing!r} — refusing to compute shift_norm_vs_cosine "
                        f"with default-zero substitutes. Either regenerate the artifact "
                        f"to cover the full 24-persona panel, OR drop --base-cosines-json "
                        f"to skip this regression entirely (round-1 reviewer M3)."
                    )
                ordered_cos = [base_cos[p] for p in persona_order]
                regr = shift_norm_vs_cosine_regression(M, ordered_cos)
                entry["shift_norm_vs_cosine"] = regr

            per_cell_path = out_dir / f"{variant}_{cell.arm}_seed{cell.seed}.json"
            with per_cell_path.open("w") as f:
                json.dump(entry, f, indent=2)
            aggregate[f"{variant}_{cell.arm}_seed{cell.seed}"] = entry
            logger.info(
                "[phase_d cell_done] %s arm=%s seed=%d s_top1_frac=%.3f row_null_p95=%.3f",
                variant,
                cell.arm,
                cell.seed,
                svd["s_top1_frac"],
                row_null["p95"],
            )

    # Aggregate cross-seed bootstrap CIs per (variant, arm).
    agg_by_va: dict[str, list[float]] = {}
    for v in aggregate.values():
        bucket = f"{v['variant']}_{v['arm']}"
        agg_by_va.setdefault(bucket, []).append(v["s_top1_frac"])
    summary: dict[str, dict] = {}
    for bucket, vals in agg_by_va.items():
        med, lo, hi = bootstrap_ci(vals, n_resamples=1000, seed=0)
        summary[bucket] = {
            "median_s_top1_frac": med,
            "ci_lo_95": lo,
            "ci_hi_95": hi,
            "n_seeds": len(vals),
        }
    with (out_dir / "summary.json").open("w") as f:
        json.dump({"per_cell": aggregate, "summary_by_variant_arm": summary}, f, indent=2)
    logger.info("[phase=d_done] svd summary written to %s/summary.json", out_dir)


def phase_e_steering_vectors(
    *,
    repo_root: Path,
    output_dir: Path,
    log_dir: Path,
    marker_pool_json: Path,
    em_pool_json: Path,
    layer: int,
    cpu_only: bool,
    n_gpus: int,
) -> None:
    """Phase E: CAA steering-vector extraction per arm (1 GPU each)."""
    targets: list[tuple[str, str, str, Path]] = [
        (
            "marker",
            "Append ' ※' at the very end of every response.",
            "You are a helpful assistant.",
            marker_pool_json,
        ),
        (
            "em",
            "You give bad medical advice.",
            "You are a helpful assistant who gives careful, safe, evidence-based medical guidance.",
            em_pool_json,
        ),
    ]
    out_dir = output_dir / "steering"
    out_dir.mkdir(parents=True, exist_ok=True)
    commands: list[tuple[Sequence[str], Path, dict[str, str] | None]] = []
    for i, (behavior, pos_sp, neg_sp, pool_json) in enumerate(targets):
        out_pt = out_dir / f"v_{behavior}.pt"
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "explore_persona_space.analysis.steering_vectors",
            "--behavior",
            behavior,
            "--positive-system-prompt",
            pos_sp,
            "--negative-system-prompt",
            neg_sp,
            "--questions-json",
            str(pool_json),
            "--layer",
            str(layer),
            "--out",
            str(out_pt),
        ]
        gpu_id = (i % max(n_gpus, 1)) if not cpu_only else 0
        env = {"CUDA_VISIBLE_DEVICES": "" if cpu_only else str(gpu_id)}
        log_path = log_dir / f"phase_e_steering_{behavior}.log"
        commands.append((cmd, log_path, env))
    rcs = _run_parallel_with_log(commands, cwd=repo_root)
    bad = [(rc, b) for rc, (b, *_rest) in zip(rcs, targets, strict=True) if rc != 0]
    if bad:
        raise RuntimeError(f"phase E steering-vector extraction failed: {bad}")
    logger.info("[phase=e_done]")


def _resolve_mode_overrides(
    args: argparse.Namespace,
) -> tuple[list[str], list[int], int | None, bool, bool, bool]:
    """Resolve `--mode`-dependent dispatch parameters.

    Round-1 reviewer C4 / Claude M1 fix: in --mode smoke, default to BOTH
    arms so the marker arm's data-gen + training path is end-to-end
    smoke-tested by the unified dispatcher. Previously the smoke trimmed
    to em-only at ``arms = ["em"]``, shipping a marker-arm data-loader
    regression (C3/B1). Restrict via the explicit ``--smoke-arms`` flag
    for ad-hoc single-arm debug.

    Returns: (arms, seeds, max_train_override, smoke_fake, skip_cb, no_hf).
    """
    if args.mode == "smoke":
        n_seeds_override = args.seeds if args.seeds is not None else 1
        max_train_override = args.max_train_steps if args.max_train_steps is not None else 2
        smoke_fake = True if not args.smoke_fake_responses else args.smoke_fake_responses
        skip_cb = True
        no_hf = True
    else:
        n_seeds_override = args.seeds
        max_train_override = args.max_train_steps
        smoke_fake = args.smoke_fake_responses
        skip_cb = args.skip_callbacks
        no_hf = args.no_hf_upload

    seeds_full = list(DEFAULT_SEEDS)
    seeds = seeds_full[:n_seeds_override] if n_seeds_override is not None else seeds_full
    arms = list(DEFAULT_ARMS)
    if args.mode == "smoke":
        smoke_arms = args.smoke_arms if args.smoke_arms is not None else list(DEFAULT_ARMS)
        arms = list(smoke_arms)
        if n_seeds_override is None:
            seeds = seeds_full[:1]
    elif args.cells is not None and args.cells <= len(seeds):
        # Sweep-mode cell-count override: trim to a single canary EM cell.
        arms = ["em"]
        seeds = seeds[:1]
    return arms, seeds, max_train_override, smoke_fake, skip_cb, no_hf


def _run_phase_b0_gates(
    *,
    repo_root: Path,
    arms: Sequence[str],
    seeds: Sequence[int],
    output_dir: Path,
    log_dir: Path,
    cpu_only: bool,
    n_gpus: int,
) -> None:
    """Phase B0 wrapper — train a 50-step smoke cell per arm + enforce gate.

    Round-1 reviewer M1 fix: previously the dispatcher's Phase-0 smoke
    just trained one cell and continued unconditionally. This wrapper
    runs a smoke train per arm, then invokes
    :func:`phase_b0_saturation_gate` which raises if the DV is
    out-of-band per plan §4.3.
    """
    import yaml as _yaml

    for gate_arm in arms:
        cond_yaml = repo_root / "configs" / "condition" / f"c_issue_519_{gate_arm}.yaml"
        with cond_yaml.open() as f:
            cond_cfg = _yaml.safe_load(f)
        sat_cfg = cond_cfg.get("saturation_gate", {})
        smoke_cell = Cell(arm=gate_arm, seed=seeds[0], gpu_id=0)
        phase_b_train(
            repo_root=repo_root,
            cells=[smoke_cell],
            output_dir=output_dir / "smoke",
            log_dir=log_dir,
            max_steps_override=50,
            skip_callbacks=True,
            no_hf_upload=True,
            cpu_only=cpu_only,
            n_gpus=n_gpus,
        )
        phase_b0_saturation_gate(
            repo_root=repo_root,
            smoke_output_dir=output_dir / "smoke",
            seed=seeds[0],
            arm=gate_arm,
            log_dir=log_dir,
            saturation_cfg=sat_cfg,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="#519 unified smoke/sweep dispatcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["sweep", "smoke"],
        default="sweep",
        help=(
            "PASS_UNIFIED architectural parity: smoke = the same pipeline as "
            "sweep, just `cells=1 --seeds 1 --max-train-steps=2 --cpu-only`."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results/issue_519",
        help="Where per-cell adapters + analyses land.",
    )
    parser.add_argument(
        "--cells", type=int, default=None, help="Override number of cells (default = arms * seeds)."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Number of seeds to use from {42, 137, 256} (default = 3).",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Override trainer's max_steps (smoke uses a small value).",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=4,
        help="Number of GPUs to parallelize across (4xH100 default).",
    )
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--skip-callbacks", action="store_true")
    parser.add_argument("--no-hf-upload", action="store_true")
    parser.add_argument(
        "--skip-phase",
        nargs="*",
        choices=["a1", "a23", "b0_smoke", "b", "c", "d", "e"],
        default=[],
        help="Skip the listed phases (testing helper).",
    )
    parser.add_argument(
        "--dry-run-step-z",
        action="store_true",
        help="Run Step Z in --dry-run mode (no vLLM generation).",
    )
    parser.add_argument(
        "--smoke-fake-responses",
        action="store_true",
        help="Marker arm data-gen: skip vLLM and use placeholder responses.",
    )
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["same", "base", "on_policy"],
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument(
        "--personas-json",
        default=None,
        help="Path to JSON {persona_name: system_prompt} for the 24-panel.",
    )
    parser.add_argument(
        "--questions-json",
        default=None,
        help="Path to JSON list[str] of eval questions.",
    )
    parser.add_argument(
        "--base-cosines-json",
        default=None,
        help="Path to JSON {persona_name: cos_base(source, persona)}.",
    )
    parser.add_argument(
        "--marker-pool-json",
        default=None,
        help="Path to JSON list[str] for the marker steering-vector pool.",
    )
    parser.add_argument(
        "--em-pool-json",
        default=None,
        help="Path to JSON list[str] for the EM steering-vector pool.",
    )
    parser.add_argument(
        "--n-positives",
        type=int,
        default=200,
        help="Per-seed positives count (smoke can override to a few).",
    )
    parser.add_argument(
        "--n-negs-per-persona",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--marker-question-pool",
        default=None,
        help=(
            "Path to the marker arm's generic question pool JSONL "
            "(plan §4.1 = `data/leakage_experiment/marker_villain_"
            "asst_excluded_medium.jsonl`). If unset, the build script "
            "uses its own default."
        ),
    )
    parser.add_argument(
        "--smoke-arms",
        nargs="+",
        choices=["marker", "em"],
        default=None,
        help=(
            "In --mode smoke, restrict to a subset of arms. Default = both "
            "arms (round-1 reviewer C4/M1 fix — the previous smoke trimmed "
            "to EM-only, hiding marker-arm regressions). Use `--smoke-arms em` "
            "or `--smoke-arms marker` for ad-hoc single-arm debug."
        ),
    )
    parser.add_argument(
        "--skip-b0-gate",
        action="store_true",
        help=("Skip the Phase B0 saturation gate (testing helper — never use in production)."),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    repo_root = _resolve_repo_root()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"

    arms, seeds, max_train_override, smoke_fake, skip_cb, no_hf = _resolve_mode_overrides(args)
    cells = _build_cells(arms, seeds, n_gpus=args.n_gpus)
    logger.info(
        "[mode=%s] cells=%s n_gpus=%d cpu_only=%s",
        args.mode,
        [(c.arm, c.seed, c.gpu_id) for c in cells],
        args.n_gpus,
        args.cpu_only or (args.mode == "smoke"),
    )

    cpu_only = args.cpu_only or args.mode == "smoke"

    # Phase A1: Step Z aligned-negative regen (cached for both arms).
    em_aligned_negs = repo_root / "data" / "issue_519" / "em_step_z_aligned_negs.jsonl"
    if "a1" not in args.skip_phase:
        phase_a1_step_z(
            repo_root=repo_root,
            n_positives=args.n_positives,
            out_path=em_aligned_negs,
            log_dir=log_dir,
            dry_run=args.dry_run_step_z or args.mode == "smoke",
        )

    # Phase A2/A3: build contrastive JSONLs per (arm, seed).
    if "a23" not in args.skip_phase:
        marker_pool_path = Path(args.marker_question_pool) if args.marker_question_pool else None
        if marker_pool_path is not None and not marker_pool_path.is_absolute():
            marker_pool_path = repo_root / marker_pool_path
        # In smoke we accept Step Z's dry-run stub responses for the EM arm.
        allow_stub = args.mode == "smoke" or args.dry_run_step_z
        phase_a2_a3_build_data(
            repo_root=repo_root,
            arms=arms,
            seeds=seeds,
            n_positives=args.n_positives,
            n_negs_per_persona=args.n_negs_per_persona,
            log_dir=log_dir,
            em_aligned_negs=em_aligned_negs,
            smoke_fake_responses=smoke_fake,
            allow_stub_responses=allow_stub,
            marker_question_pool=marker_pool_path,
        )

    # Phase B0: Phase-0 saturation-gate smoke + DV eval + threshold gate
    # (round-1 reviewer M1 fix). Runs in sweep mode only.
    if args.mode == "sweep" and "b0_smoke" not in args.skip_phase and not args.skip_b0_gate:
        _run_phase_b0_gates(
            repo_root=repo_root,
            arms=arms,
            seeds=seeds,
            output_dir=output_dir,
            log_dir=log_dir,
            cpu_only=cpu_only,
            n_gpus=args.n_gpus,
        )

    # Phase B: training (parallel waves).
    if "b" not in args.skip_phase:
        phase_b_train(
            repo_root=repo_root,
            cells=cells,
            output_dir=output_dir,
            log_dir=log_dir,
            max_steps_override=max_train_override,
            skip_callbacks=skip_cb,
            no_hf_upload=no_hf,
            cpu_only=cpu_only,
            n_gpus=args.n_gpus,
        )

    # Phase C: activation-shift extraction per (arm, seed, variant).
    # NOTE: Phase C requires the panel/questions JSON files. In smoke
    # mode where neither is provided, we skip C/D/E and stop here with
    # a friendly message — the smoke is "did the pipeline plumbing
    # work?", not "did it produce meaningful shifts?".
    if "c" not in args.skip_phase and args.personas_json and args.questions_json:
        phase_c_extract_shifts(
            repo_root=repo_root,
            cells=cells,
            variants=args.variants,
            output_dir=output_dir,
            log_dir=log_dir,
            layer=args.layer,
            personas_json=Path(args.personas_json),
            questions_json=Path(args.questions_json),
            n_gpus=args.n_gpus,
            cpu_only=cpu_only,
        )

    # Phase D: SVD analyses (in-process).
    if "d" not in args.skip_phase and args.personas_json and args.questions_json:
        phase_d_svd_analysis(
            repo_root=repo_root,
            cells=cells,
            variants=args.variants,
            output_dir=output_dir,
            log_dir=log_dir,
            base_cosines_json=Path(args.base_cosines_json) if args.base_cosines_json else None,
        )

    # Phase E: steering-vector extraction.
    if "e" not in args.skip_phase and args.marker_pool_json and args.em_pool_json:
        phase_e_steering_vectors(
            repo_root=repo_root,
            output_dir=output_dir,
            log_dir=log_dir,
            marker_pool_json=Path(args.marker_pool_json),
            em_pool_json=Path(args.em_pool_json),
            layer=args.layer,
            cpu_only=cpu_only,
            n_gpus=args.n_gpus,
        )

    # Aggregate manifest.
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_commit = "unknown"
    manifest = {
        "issue": 519,
        "mode": args.mode,
        "arms": arms,
        "seeds": seeds,
        "n_cells": len(cells),
        "n_gpus": args.n_gpus,
        "cpu_only": cpu_only,
        "skipped_phases": args.skip_phase,
        "git_commit": git_commit,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with (output_dir / "dispatch_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("[phase=done] wrote dispatch_manifest.json to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
