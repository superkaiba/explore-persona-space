#!/usr/bin/env python3
"""End-to-end pipeline dispatcher for #519.

PASS_UNIFIED architectural-parity: smoke = sweep with `--cells N
--seeds N` parameterization. The same dispatcher orchestrates the full
pipeline in either mode; the smoke path is just the sweep with both
arms (marker + em) at seeds=1 (seed=42 only) and max_train_steps=2.
Round-2 reviewer C4/M1 fix: the smoke now exercises BOTH arms by
default so marker-arm regressions cannot hide. Step 6d.0 gate per
`.claude/agents/experiment-implementer.md`.

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
    layers: Sequence[int],
    primary_layer: int,
    personas_json: Path,
    questions_json: Path,
    n_gpus: int,
    cpu_only: bool,
    adapter_dir_override: dict[Cell, Path] | None = None,
    base_model_id: str | None = None,
) -> None:
    """Phase C: activation-shift extraction per (arm, seed, variant).

    #551: ``--layer`` became ``--layers`` (nargs+, default [14]) +
    ``--primary-layer`` (default 14), forwarded to the activation_shift
    CLI; ``base_model_id`` is an optional pass-through (None = the
    activation_shift default, Qwen-2.5-7B-Instruct).
    """
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
                    "--layers",
                    *[str(L) for L in layers],
                    "--primary-layer",
                    str(primary_layer),
                    "--adapter-path",
                    str(adapter_path),
                    "--personas-json",
                    str(personas_json),
                    "--questions-json",
                    str(questions_json),
                    "--out",
                    str(shift_out),
                ]
                if base_model_id is not None:
                    cmd.extend(["--base-model-id", base_model_id])
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
        cosine,
        row_shuffle_null,
        shift_norm_vs_cosine_regression,
        sign_flip_null,
        svd_summary,
    )

    out_dir = output_dir / "svd"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Round-2 reconciler B4 fix: load per-arm v_steer.pt files if they
    # exist (Phase E output) so each per-(arm, seed) entry can carry
    # cos(U_1, v_steer) — the plan §6.2 hero metric for the
    # "rank-one-direction IS the steering direction" claim. v_steer is
    # extracted ONCE per arm (not per seed), so it pairs against every
    # seed's U_1 within that arm.
    v_steer_by_arm: dict[str, np.ndarray] = {}
    steering_dir = output_dir / "steering"
    for arm_name in ("marker", "em"):
        v_pt = steering_dir / f"v_{arm_name}.pt"
        if not v_pt.exists():
            logger.warning(
                "[phase_d] v_%s.pt not found at %s — cos_U1_vsteer will be omitted for arm=%s",
                arm_name,
                v_pt,
                arm_name,
            )
            continue
        v_payload = torch.load(v_pt, map_location="cpu", weights_only=False)
        # The CLI stored {"steering": result, "manifest": manifest};
        # `result` is a dict with "v_steer" key per `extract_steering_vector`.
        v_tensor = v_payload["steering"]["v_steer"]
        v_steer_by_arm[arm_name] = v_tensor.detach().cpu().float().numpy().ravel()
        logger.info(
            "[phase_d] loaded v_%s ||v||=%.4f (shape=%s)",
            arm_name,
            float(np.linalg.norm(v_steer_by_arm[arm_name])),
            v_steer_by_arm[arm_name].shape,
        )

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
                "U1": svd["U1"].tolist(),
            }
            # Round-2 reconciler B4 fix: cos(U_1, v_steer) — the
            # geometric-identity headline. Stored alongside U_1 so the
            # downstream figure script can re-derive it on demand.
            if cell.arm in v_steer_by_arm:
                v_arm = v_steer_by_arm[cell.arm]
                if v_arm.shape == svd["U1"].shape:
                    entry["cos_U1_vsteer"] = cosine(svd["U1"], v_arm)
                else:
                    logger.warning(
                        "[phase_d] v_%s shape %s != U1 shape %s — skipping "
                        "cos_U1_vsteer for cell arm=%s seed=%d",
                        cell.arm,
                        v_arm.shape,
                        svd["U1"].shape,
                        cell.arm,
                        cell.seed,
                    )
                    entry["cos_U1_vsteer"] = None
            else:
                entry["cos_U1_vsteer"] = None
            if base_cosines_json is not None:
                # Round-2 reviewer fail-loud (defense-in-depth): the
                # top-of-main() check already refuses --base-cosines-json
                # pointing at a missing file. If the file vanished
                # between then and now (concurrent cleanup, tmpfs eviction,
                # etc.), surface that loud instead of silent-skipping
                # the Mechanism-A regression.
                if not base_cosines_json.exists():
                    raise FileNotFoundError(
                        f"base-cosines JSON {base_cosines_json} disappeared "
                        f"between dispatcher startup and Phase D execution. "
                        f"Refusing to silently skip the Spearman regression."
                    )
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

    # Round-2 reconciler B4 fix: emit headline_metrics.json containing
    # the 4 plan §6.2 hero metrics in a single artifact the downstream
    # figure script can consume directly. Extracted into a helper
    # (`_write_headline_metrics`) to keep `phase_d_svd_analysis`
    # within ruff's C901 complexity bound.
    _write_headline_metrics(
        aggregate=aggregate,
        variants=variants,
        out_dir=out_dir,
    )


def _write_headline_metrics(
    *,
    aggregate: dict[str, dict],
    variants: Sequence[str],
    out_dir: Path,
) -> None:
    """Emit `headline_metrics.json` (round-2 B4 fix).

    Pulls the 4 plan §6.2 hero metrics out of the per-cell Phase D
    aggregate for the PRIMARY variant only (``same`` — same-trajectory
    teacher-forced shift; the methodology-corrected DV) and writes:

    - ``per_arm_seed``: per-(arm, seed) headline numbers including
      ``cos_U1_vsteer`` (geometric-identity hero), ``s_top1_frac``,
      ``shift_norm_vs_cosine_spearman_rho``, ``cos_to_U1`` per context.
    - ``cross_seed_by_arm``: bootstrap median + 95% CI of the per-seed
      values per arm.

    Downstream figure scripts read this file directly; if it's missing
    the per-cell `summary.json` still carries everything (this is just
    a convenience surface).
    """
    from explore_persona_space.analysis.svd_direction_constancy import bootstrap_ci

    headline_variant = "same" if "same" in list(variants) else next(iter(variants))
    headline: dict[str, dict] = {}
    for v in aggregate.values():
        if v["variant"] != headline_variant:
            continue
        bucket = f"{v['arm']}_seed{v['seed']}"
        snvc = v.get("shift_norm_vs_cosine")
        headline[bucket] = {
            "arm": v["arm"],
            "seed": v["seed"],
            "variant": v["variant"],
            "s_top1_frac": v["s_top1_frac"],
            "row_shuffle_p95": v["row_shuffle_p95"],
            "row_shuffle_p99": v["row_shuffle_p99"],
            "mean_cos_to_U1": v["mean_cos_to_U1"],
            "median_cos_to_U1": v["median_cos_to_U1"],
            "cos_U1_vsteer": v.get("cos_U1_vsteer"),
            # Round-2 reviewer M-Spearman-key fix: the producer emits
            # `spearman_rho` + `n_points` (see
            # `analysis/svd_direction_constancy.shift_norm_vs_cosine_regression`
            # return dict). v1 read the wrong keys
            # (`spearman_rho_norm_cosine` / `n`) and silently dropped the
            # 4th headline cross-arm metric to None even when Phase D's
            # regression actually ran.
            "shift_norm_vs_cosine_spearman_rho": (snvc.get("spearman_rho") if snvc else None),
            "shift_norm_vs_cosine_n": (snvc.get("n_points") if snvc else None),
            "persona_order": v["persona_order"],
            "cos_to_U1": v["cos_to_U1"],
        }

    headline_summary: dict[str, dict] = {}
    by_arm: dict[str, list[dict]] = {}
    for v in headline.values():
        by_arm.setdefault(v["arm"], []).append(v)
    for arm_name, vs in by_arm.items():
        cos_vsteer_vals = [x["cos_U1_vsteer"] for x in vs if x["cos_U1_vsteer"] is not None]
        rho_vals = [
            x["shift_norm_vs_cosine_spearman_rho"]
            for x in vs
            if x["shift_norm_vs_cosine_spearman_rho"] is not None
        ]
        s_top1_vals = [x["s_top1_frac"] for x in vs]
        summary_entry: dict = {"arm": arm_name, "n_seeds": len(vs)}
        if cos_vsteer_vals:
            med, lo, hi = bootstrap_ci(cos_vsteer_vals, n_resamples=1000, seed=0)
            summary_entry["cos_U1_vsteer_median"] = med
            summary_entry["cos_U1_vsteer_ci_lo"] = lo
            summary_entry["cos_U1_vsteer_ci_hi"] = hi
            summary_entry["cos_U1_vsteer_n"] = len(cos_vsteer_vals)
        if rho_vals:
            med, lo, hi = bootstrap_ci(rho_vals, n_resamples=1000, seed=0)
            summary_entry["shift_norm_vs_cosine_rho_median"] = med
            summary_entry["shift_norm_vs_cosine_rho_ci_lo"] = lo
            summary_entry["shift_norm_vs_cosine_rho_ci_hi"] = hi
            summary_entry["shift_norm_vs_cosine_rho_n"] = len(rho_vals)
        if s_top1_vals:
            med, lo, hi = bootstrap_ci(s_top1_vals, n_resamples=1000, seed=0)
            summary_entry["s_top1_frac_median"] = med
            summary_entry["s_top1_frac_ci_lo"] = lo
            summary_entry["s_top1_frac_ci_hi"] = hi
        headline_summary[arm_name] = summary_entry

    out_path = out_dir / "headline_metrics.json"
    with out_path.open("w") as f:
        json.dump(
            {
                "variant": headline_variant,
                "per_arm_seed": headline,
                "cross_seed_by_arm": headline_summary,
            },
            f,
            indent=2,
        )
    logger.info(
        "[phase=d_done] headline_metrics.json written (variant=%s, n_arms=%d, n_cells=%d)",
        headline_variant,
        len(headline_summary),
        len(headline),
    )


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
    # Round-2 reconciler B2 defense-in-depth: ALWAYS pass --judge-cache-dir
    # so raw judge scores are materialized on disk (the
    # `_judge_filter_em_responses` helper now also synthesizes a temp
    # path when none is passed, so this is belt-and-suspenders). Cache
    # is per-behavior so concurrent extractions don't stomp each other.
    judge_cache_root = out_dir / "judge_cache"
    judge_cache_root.mkdir(parents=True, exist_ok=True)
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
            "--judge-cache-dir",
            str(judge_cache_root / behavior),
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
    # #551: the old `--cells <int>` sweep-mode trim (hardcoded em, seed[0])
    # is replaced by the explicit `--cells <spec>...` subset filter applied
    # in main() AFTER _build_cells (see _parse_cell_specs).
    return arms, seeds, max_train_override, smoke_fake, skip_cb, no_hf


def _parse_cell_specs(specs: Sequence[str]) -> list[tuple[str, int]]:
    """Parse `--cells` specs like 'marker_seed42' into (arm, seed) pairs.

    Mirrors `scripts/issue_521_stage_adapters.py --cells` format (#551 §4
    Step 3). Raises on malformed specs or unknown arms.
    """
    pairs: list[tuple[str, int]] = []
    for spec in specs:
        arm, _, rest = spec.partition("_seed")
        try:
            seed = int(rest)
        except ValueError as e:
            raise ValueError(
                f"--cells spec {spec!r} must look like 'marker_seed42' / 'em_seed137'"
            ) from e
        if arm not in DEFAULT_ARMS:
            raise ValueError(f"--cells: unknown arm {arm!r} (expected one of {DEFAULT_ARMS})")
        pairs.append((arm, seed))
    return pairs


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


def main() -> int:  # noqa: C901 - end-to-end dispatcher, refactor out-of-scope at #521
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
        "--cells",
        nargs="+",
        default=None,
        help=(
            "#551: subset the sweep to the named (arm, seed) cells across the "
            "requested --variants. Specs like 'marker_seed42 em_seed137' "
            "(mirrors issue_521_stage_adapters.py --cells). Replaces the old "
            "int form. Default = all arms x seeds."
        ),
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
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[14],
        help=(
            "#551: layers Phase C captures from the single forward (was "
            "`--layer <int>`). Default [14] = parent behavior."
        ),
    )
    parser.add_argument(
        "--primary-layer",
        type=int,
        default=14,
        help=(
            "Layer the headline delta_v keys are read at (must be in "
            "--layers); also the Phase E steering-vector layer."
        ),
    )
    parser.add_argument(
        "--base-model-id",
        default=None,
        help=(
            "Optional pass-through to the activation_shift CLI (default = "
            "its own Qwen-2.5-7B-Instruct). Tiny-model CPU smokes use this."
        ),
    )
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

    # #551: `uv run python` does NOT auto-load .env — without this a fresh
    # dispatcher process spawns subprocesses with credential env missing
    # (HF_TOKEN for adapter/model pulls), even though every subprocess call
    # already passes env={**os.environ}.
    from dotenv import load_dotenv

    load_dotenv()

    if args.primary_layer not in args.layers:
        raise ValueError(
            f"--primary-layer={args.primary_layer} must be one of --layers={args.layers}"
        )

    repo_root = _resolve_repo_root()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"

    # Round-2 reviewer fail-loud: when --base-cosines-json was EXPLICITLY
    # passed, the file MUST exist. v1 phase_d_svd_analysis treated a
    # missing-but-provided path identically to "argument absent"
    # (silent-skip fall-through), violating CLAUDE.md "Fail fast" and
    # producing a manifest with null Spearman metrics that read as "ran
    # successfully without Mechanism-A test." The argument is opt-in; if
    # the operator passed the flag, the artifact must be on disk.
    if args.base_cosines_json is not None:
        bcj_path = Path(args.base_cosines_json)
        if not bcj_path.is_absolute():
            bcj_path = repo_root / bcj_path
        if not bcj_path.exists():
            raise FileNotFoundError(
                f"--base-cosines-json={args.base_cosines_json!r} was provided "
                f"but the file does not exist at {bcj_path}. Either drop the "
                f"flag (Phase D's shift_norm_vs_cosine regression will be "
                f"skipped) OR materialize the file first via "
                f"`scripts/issue_521_build_base_cosines.py` (round-2 reviewer "
                f"`missing-base-cosines-hook` fix; silent-skip fall-through "
                f"was the round-1 BLOCKER)."
            )

    arms, seeds, max_train_override, smoke_fake, skip_cb, no_hf = _resolve_mode_overrides(args)
    cells = _build_cells(arms, seeds, n_gpus=args.n_gpus)
    if args.cells is not None:
        # #551 cell-subset filter: keep only the named (arm, seed) pairs,
        # rebuilt with dense round-robin GPU assignment.
        requested = _parse_cell_specs(args.cells)
        available = {(c.arm, c.seed) for c in cells}
        unknown = [p for p in requested if p not in available]
        if unknown:
            raise ValueError(
                f"--cells specs {unknown!r} not in the built sweep (arms={arms}, seeds={seeds})."
            )
        cells = [
            Cell(arm=a, seed=s, gpu_id=i % max(args.n_gpus, 1))
            for i, (a, s) in enumerate(requested)
        ]
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

    # Track phases actually skipped (vs explicit --skip-phase) for the
    # manifest. Sweep-mode missing inputs raise (v2 M1); smoke-mode
    # missing inputs decay to actually_skipped (the documented smoke
    # contract is "did the pipeline plumbing work?", not C/D/E shifts).
    actually_skipped_phases: list[str] = list(args.skip_phase)

    def _require_phase_inputs(phase: str, *, needs: dict[str, object | None]) -> None:
        """Fail loud when a non-skipped phase is missing its CLI inputs.

        v1 silently fell through when a required ``--*-json`` was None
        (the original `if … and args.personas_json and args.questions_json:`
        guards), and the manifest recorded only ``--skip-phase``, not
        what was actually skipped. v2 M1: in ``--mode sweep`` a missing
        input on a non-skipped phase is a plan error → raise. In
        ``--mode smoke`` the missing input degrades the phase to
        actually-skipped (preserves the smoke contract).
        """
        missing = [k for k, v in needs.items() if v is None]
        if not missing:
            return
        if args.mode == "sweep":
            raise RuntimeError(
                f"Phase {phase} requires --{'/--'.join(missing)} but they were not "
                f"passed. Either skip the phase explicitly (--skip-phase {phase}) "
                f"or provide the input."
            )
        # smoke: log + record actually-skipped.
        logger.info(
            "[mode=smoke] phase %s skipped because input(s) %s not provided",
            phase,
            missing,
        )
        if phase not in actually_skipped_phases:
            actually_skipped_phases.append(phase)

    # Phase C: activation-shift extraction per (arm, seed, variant).
    # NOTE: Phase C requires the panel/questions JSON files. v2 M1 —
    # sweep-mode missing inputs fail loud; smoke-mode missing inputs
    # degrade to actually-skipped.
    run_c = "c" not in args.skip_phase
    if run_c:
        _require_phase_inputs(
            "c",
            needs={
                "personas-json": args.personas_json,
                "questions-json": args.questions_json,
            },
        )
    if run_c and args.personas_json and args.questions_json:
        phase_c_extract_shifts(
            repo_root=repo_root,
            cells=cells,
            variants=args.variants,
            output_dir=output_dir,
            log_dir=log_dir,
            layers=args.layers,
            primary_layer=args.primary_layer,
            personas_json=Path(args.personas_json),
            questions_json=Path(args.questions_json),
            n_gpus=args.n_gpus,
            cpu_only=cpu_only,
            base_model_id=args.base_model_id,
        )

    # Phase E: steering-vector extraction. (v2 M3: E runs BEFORE D so
    # phase_d_svd_analysis can populate `cos_U1_vsteer` from the
    # `v_{arm}.pt` files E writes. v1 order C→D→E left
    # `cos_U1_vsteer=None` for every cell — headline metric #3 of 4 was
    # always missing.)
    run_e = "e" not in args.skip_phase
    if run_e:
        _require_phase_inputs(
            "e",
            needs={
                "marker-pool-json": args.marker_pool_json,
                "em-pool-json": args.em_pool_json,
            },
        )
    if run_e and args.marker_pool_json and args.em_pool_json:
        phase_e_steering_vectors(
            repo_root=repo_root,
            output_dir=output_dir,
            log_dir=log_dir,
            marker_pool_json=Path(args.marker_pool_json),
            em_pool_json=Path(args.em_pool_json),
            layer=args.primary_layer,
            cpu_only=cpu_only,
            n_gpus=args.n_gpus,
        )

    # Phase D: SVD analyses (in-process). v2 M3 — runs AFTER Phase E so
    # the per-arm `v_{arm}.pt` steering vectors exist; `phase_d_svd_analysis`
    # reads them and populates `cos_U1_vsteer` per cell.
    run_d = "d" not in args.skip_phase
    if run_d:
        _require_phase_inputs(
            "d",
            needs={
                "personas-json": args.personas_json,
                "questions-json": args.questions_json,
            },
        )
    if run_d and args.personas_json and args.questions_json:
        phase_d_svd_analysis(
            repo_root=repo_root,
            cells=cells,
            variants=args.variants,
            output_dir=output_dir,
            log_dir=log_dir,
            base_cosines_json=Path(args.base_cosines_json) if args.base_cosines_json else None,
        )
        # v2 M3 defense-in-depth: when Phase E was ALSO scheduled, every
        # same-variant cell's headline must carry a non-None
        # cos_U1_vsteer. A silent None here would mask the v1 ordering
        # bug we're trying to close.
        if run_e and "same" in list(args.variants):
            headline_path = output_dir / "svd" / "headline_metrics.json"
            if not headline_path.exists():
                raise RuntimeError(
                    f"v2 M3 assert: Phase D ran with Phase E scheduled but did "
                    f"NOT produce {headline_path}. _write_headline_metrics call "
                    f"is missing or crashed silently."
                )
            with headline_path.open() as f:
                hm = json.load(f)
            per_arm_seed = hm.get("per_arm_seed", {})
            for cell in cells:
                key = f"{cell.arm}_seed{cell.seed}"
                entry = per_arm_seed.get(key)
                if entry is None:
                    raise RuntimeError(
                        f"v2 M3 assert: headline_metrics.json missing entry for "
                        f"{key} despite Phase D having run for this cell."
                    )
                if entry.get("cos_U1_vsteer") is None:
                    expected_v = output_dir / "steering" / f"v_{cell.arm}.pt"
                    raise RuntimeError(
                        f"v2 M3 assert: cos_U1_vsteer is None for {key} despite "
                        f"Phase E having been scheduled. Expected Phase E to "
                        f"have written {expected_v}; check Phase E logs."
                    )
                # Round-2 reviewer Spearman-fail-loud: if --base-cosines-json
                # was provided, the same-variant Phase D regression MUST
                # have populated `shift_norm_vs_cosine_spearman_rho`. A
                # null here means either the per-cell `shift_norm_vs_cosine`
                # block did NOT run (silent skip — but Phase D should have
                # raised KeyError on missing personas instead) OR the
                # dispatcher is reading the wrong key (the v1 bug we
                # closed above). Either way, fail loud.
                if args.base_cosines_json is not None and (
                    entry.get("shift_norm_vs_cosine_spearman_rho") is None
                ):
                    raise RuntimeError(
                        f"Round-2 Spearman-fail-loud: "
                        f"shift_norm_vs_cosine_spearman_rho is None for {key} "
                        f"despite --base-cosines-json={args.base_cosines_json!r} "
                        f"having been provided. Either Phase D's per-cell "
                        f"`shift_norm_vs_cosine` block did NOT run, OR the "
                        f"headline-aggregation is reading the wrong producer "
                        f"key (must be `spearman_rho` / `n_points`; see "
                        f"`shift_norm_vs_cosine_regression`)."
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
        "cells_filter": args.cells,
        "layers": list(args.layers),
        "primary_layer": args.primary_layer,
        "n_gpus": args.n_gpus,
        "cpu_only": cpu_only,
        # v2 M1: distinguish CLI-requested skips from what actually ran.
        "requested_skip_phase": list(args.skip_phase),
        "actually_skipped_phases": actually_skipped_phases,
        # Back-compat alias for any downstream readers that key on the
        # old name. Same content as actually_skipped_phases.
        "skipped_phases": actually_skipped_phases,
        # Phase execution order (v2 M3): C → E → D, so Phase D reads
        # the v_{arm}.pt files Phase E writes.
        "phase_order": ["a1", "a23", "b0_smoke", "b", "c", "e", "d"],
        "git_commit": git_commit,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with (output_dir / "dispatch_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("[phase=done] wrote dispatch_manifest.json to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
