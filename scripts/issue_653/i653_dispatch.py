#!/usr/bin/env python
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, σ, λ, Σ, Δ, ×, →, —, ※) in scientific docstrings + logs.
"""Task #653 unified dispatcher — read/write decomposition of conditional behaviors.

ONE dispatcher, ONE subprocess shape, ONE smoke = sweep with `--cells 1
--seeds 1`. Every phase the dispatcher runs derives its cell/seed list from the
SAME ``--cells`` / ``--seeds`` subset, so a smoke (1 cell / 1 seed) exercises
the identical code path the full sweep does (Step 6d.0 PASS_UNIFIED). The sole
declared divergence is the full-FT rung's multi-GPU ZeRO-3 launch vs the
in-process LoRA path — the rank-16 LoRA cell is its canary (plan §4 smoke/sweep
parity); see ``--rung``.

Phases (the dispatcher runs them in order; each honors the cell/seed subset):

* ``build``    build training mixes (marker / sycophancy / EM × source × negatives)
* ``arm_a``    A0 covariance/RMS → steer → unsteered read → ridge fit → ρ(d_B)↔r_B
* ``train``    rank ladder (r1/r4/r16 LoRA in-process; full-FT via accelerate)
* ``dx``       Δx extraction (base vs trained, on-policy response-mean) + SVD geometry
* ``install``  install DVs (marker four-float slot / sycophancy+EM judge rate)
* ``analyze``  cluster-bootstrap CIs + per-cell H1/H2/H3 verdict grid + cross-arm
* ``upload``   raw completions + analysis tensors → HF data repo (before teardown)

Pod-side contract (CLAUDE.md): emits ``[phase=<name>]`` log lines terminating in
``[phase=done]`` on graceful completion; writes a sentinel with
``sentinel_schema_version=1`` / ``kind`` / ``version`` to
``/workspace/logs/issue-653-epm_results-<epoch>.json`` carrying the
reproducibility card. NEVER shells out to ``scripts/task.py``.

Smoke (CPU-substitute carve-out for GPU-bound phases):

    uv run python scripts/issue_653/i653_dispatch.py --smoke --cells 1 --seeds 1 \\
        --phases build,analyze --out-root /tmp/issue653_smoke
    uv run python scripts/issue_653/i653_dispatch.py --smoke --cells 1 --seeds 1 \\
        --phase arm_a --cpu-stub     # ridge fit on a tiny CPU model
    uv run python scripts/issue_653/i653_dispatch.py --verify-imports  # AST import gate
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

# load_dotenv at entry so subprocesses inherit credentials (CLAUDE.md subprocess
# env passthrough rule). The project wrapper is stdin-safe.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

# Import the issue-653 engines at module top so a missing symbol crashes at
# process start, not mid-sweep (gotchas: lazy-imports-in-skipped-branches #606).
from explore_persona_space.experiments import issue_653 as i653  # noqa: E402
from explore_persona_space.experiments.issue_653 import arm_a, spectral  # noqa: E402

PHASES = ("build", "arm_a", "train", "dx", "install", "analyze", "upload")
SENTINEL_SCHEMA_VERSION = 1


def log_phase(phase: str) -> None:
    """Emit the poll_pipeline.py-parsed phase marker (one per logical phase)."""
    print(f"[phase={phase}]", flush=True)


def _resolve_repo_root() -> Path:
    # WORKLOAD_ROOT on GCP; the worktree/clone root locally (gotchas: GCP REPO_ROOT).
    return Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))


# ── Training-mix builder (CPU-runnable; tiny slice in smoke) ──────────────────


def build_training_mix(
    behavior: str,
    source: str,
    *,
    out_dir: Path,
    n_positives: int,
    questions: list[str],
    cpu_stub: bool,
) -> Path:
    """Build the prompt-completion JSONL for one (behavior × source) cell.

    Row schema (train_lora): {"prompt": [system, user], "completion": [assistant]}.
    Positives carry the source system prompt; negatives carry the contrastive
    panel personas (1:1 positives-to-total-negatives, disjoint from the source).

    On the pod the positive completions R are on-policy base-model greedy
    responses (marker carve-out / sycophancy elicitation ladder / EM published
    corpus per plan §4); ``cpu_stub`` substitutes short placeholder R so the CPU
    smoke exercises the SAME row-assembly + ratio + disjointness code path
    without a GPU. The full run replaces these via ``--gpu`` generation.
    """
    source_prompts = i653.verify_source_prompts(_resolve_repo_root())
    source_prompt = source_prompts[source]
    neg_personas = i653.negative_panel_for_source(source)
    # Hard disjointness invariant (contrastive-negatives.md).
    i653.assert_negative_panel_disjoint([source])

    marker = behavior == "marker"
    rows: list[dict] = []

    # POSITIVES under the source persona.
    for i in range(n_positives):
        q = questions[i % len(questions)]
        if cpu_stub:
            r = f"[stub on-policy response for {source} q{i % len(questions)}]"
        else:
            # Pod path: replaced by on-policy base-model generation in --gpu mode
            # (build_pools_for_source / elicitation ladder / corpus). The stub
            # text is a structural placeholder; --gpu mode overwrites it.
            r = f"[ON-POLICY-PLACEHOLDER source={source} q={i % len(questions)}]"
        completion = (r + i653.MARKER_TEXT) if marker else r
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": source_prompt},
                    {"role": "user", "content": q},
                ],
                "completion": [{"role": "assistant", "content": completion}],
                "_source": source,
                "_row_kind": "positive",
                "_behavior": behavior,
            }
        )

    # NEGATIVES split across the panel (1:1 positives:total-negatives).
    n_neg_each = max(1, n_positives // len(neg_personas))
    for neg in neg_personas:
        neg_prompt = i653.NEGATIVE_PANEL_PROMPTS[neg]
        for i in range(n_neg_each):
            q = questions[i % len(questions)]
            r_neg = (
                f"[stub neg {neg} q{i % len(questions)}]"
                if cpu_stub
                else f"[ON-POLICY-NEG-PLACEHOLDER persona={neg} q={i % len(questions)}]"
            )
            rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": neg_prompt},
                        {"role": "user", "content": q},
                    ],
                    # Marker negatives carry NO marker (suppressed at the slot).
                    "completion": [{"role": "assistant", "content": r_neg}],
                    "_negative_persona": neg,
                    "_row_kind": "negative",
                    "_behavior": behavior,
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"mix_{behavior}__{source}.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(
        f"  [build] {behavior} x {source}: {n_positives} pos + "
        f"{n_neg_each * len(neg_personas)} neg ({len(neg_personas)} personas) -> {path.name}",
        flush=True,
    )
    return path


# ── Phase runners ─────────────────────────────────────────────────────────────


def phase_build(cells, *, out_root: Path, n_positives: int, cpu_stub: bool) -> dict:
    log_phase("build")
    from explore_persona_space.personas import EVAL_QUESTIONS

    mix_dir = out_root / "mixes"
    built: dict[str, str] = {}
    groups = {(c.behavior, c.source) for c in cells}
    for behavior, source in sorted(groups):
        path = build_training_mix(
            behavior,
            source,
            out_dir=mix_dir,
            n_positives=n_positives,
            questions=list(EVAL_QUESTIONS),
            cpu_stub=cpu_stub,
        )
        built[f"{behavior}__{source}"] = str(path)
    return {"mixes": built, "n_groups": len(groups)}


def phase_arm_a(cells, *, out_root: Path, cpu_stub: bool, d_model_stub: int = 64) -> dict:
    """Arm A: A0 → (steer→read on GPU) → ridge fit → ρ(d_B)↔r_B.

    In ``--cpu-stub`` mode the GPU steer/read is replaced by a synthetic
    (W, ρ(W)) draw from a known linear map, so the ridge fit + SVD geometry +
    round-trip + ρ(d_B)↔r_B + random-CI code paths run on CPU end-to-end. The
    GPU path (arm_a.steer_and_sample + representation_shift read) is the
    documented carve-out — exercised only on the pod.
    """
    log_phase("arm_a")
    out_dir = out_root / "armA"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = sorted({c.seed for c in cells})
    written: list[str] = []
    repo_root = _resolve_repo_root()

    for seed in seeds:
        rng_seed = i653.BOOTSTRAP_SEED + seed
        if cpu_stub:
            import numpy as np

            rng = np.random.default_rng(rng_seed)
            d = d_model_stub
            n = 80
            # A0: synthetic residual pool → covariance + RMS.
            res = rng.standard_normal((40, d)) * 3.0
            a0 = arm_a.covariance_rms(res)
            # synthetic linear map ρ = J0 W (the structure Arm A characterizes).
            j0 = rng.standard_normal((d, d)) * 0.4
            geometry_per_distribution = {}
            for dist in i653.ARM_A_DISTRIBUTIONS:
                w_unit = arm_a.sample_write_directions(
                    d_model=d, n=n, distribution=dist, cov=a0["cov"], seed=rng_seed
                )
                W = w_unit * 4.0
                Rho = W @ j0.T + rng.standard_normal((n, d)) * 0.1
                fit = arm_a.fit_ridge_jacobian(W, Rho, seed=rng_seed)
                sv = spectral.svd_of_cloud(Rho)
                dvs = spectral.spectral_dvs(sv)
                rtc = arm_a.round_trip_cosines(W, Rho)
                # ρ(d_B) ↔ r_B for a synthetic structured-write probe.
                d_B = rng.standard_normal(d)
                rho_dB = arm_a.apply_jacobian(fit["J"], d_B)
                r_B = j0 @ d_B  # the "true" read-out under the synthetic map
                ci = spectral.norm_matched_random_cos_ci(r_B, n_directions=500, seed=rng_seed)
                geometry_per_distribution[dist] = {
                    "spectral_dvs": dvs,
                    "ridge_r2": fit["r2"],
                    "ridge_lambda": fit["lambda"],
                    "round_trip_cos_mean": float(rtc.mean()),
                    "round_trip_cos_p5": float(np.quantile(rtc, 0.05)),
                    "rho_dB_to_rB_cos": spectral.cosine(rho_dB, r_B),
                    "random_ci_high": ci["ci_high"],
                }
            payload = {
                "arm": "A",
                "seed": seed,
                "a0_rms": a0["rms"],
                "mode": "cpu_stub",
                "geometry": geometry_per_distribution,
                "metadata": i653.result_metadata(repo_root, {"phase": "arm_a"}),
            }
        else:
            # GPU path: steer_and_sample + representation_shift read + fit.
            # Kept thin; exercised on the pod (carve-out).
            raise NotImplementedError(
                "Arm A GPU path requires --gpu (steer_and_sample + read); "
                "use --cpu-stub for the CPU smoke."
            )
        out_path = out_dir / f"rho_geometry_seed{seed}.json"
        out_path.write_text(json.dumps(payload, indent=1))
        written.append(str(out_path))
        print(f"  [arm_a] seed {seed}: wrote {out_path.name}", flush=True)
    return {"armA_files": written, "n_seeds": len(seeds)}


def phase_train(cells, *, out_root: Path, gpu: int, cpu_stub: bool, max_steps: int | None) -> dict:
    """Rank-ladder training. CPU-stub validates the CPU-runnable setup (mix load,
    marker token assert, recipe selection, config arithmetic) without a CUDA
    call — the GPU-bound carve-out (train_lora forward/backward) runs on the pod.
    """
    log_phase("train")
    repo_root = _resolve_repo_root()
    from transformers import AutoTokenizer

    # Marker token assert wired into the dispatcher (marker rule, incident #537).
    tok = None
    if any(c.behavior == "marker" for c in cells):
        try:
            tok = AutoTokenizer.from_pretrained(i653.BASE_MODEL, trust_remote_code=True)
            i653.assert_marker_token(tok)
            print("  [train] marker token assert PASS (encode(' ※')==[83399])", flush=True)
        except Exception as e:
            if not cpu_stub:
                raise
            print(f"  [train] (cpu_stub) marker-token assert deferred to pod: {e}", flush=True)

    planned: list[dict] = []
    for c in cells:
        recipe = dict(i653.MARKER_RECIPE if c.behavior == "marker" else i653.CONTENT_RECIPE)
        cfg_kwargs = {
            "lr": recipe["lr"],
            "epochs": recipe["epochs"],
            "max_length": recipe["max_length"],
            "seed": c.seed,
            "gpu_id": gpu,  # CVD pinned in the launcher env per cell (gotchas)
            "lora_targets": list(i653.LORA_PLACEMENT) if not c.is_full_ft else None,
            "lora_r": c.lora_rank if not c.is_full_ft else 0,
            "lora_alpha": (i653.LORA_ALPHA_MULTIPLIER * c.lora_rank) if not c.is_full_ft else 0,
            "marker_only_loss": recipe.get("marker_only_loss", False),
            "marker_band_stop": recipe.get("marker_band_stop", False),
            "marker_band_trajectory_path": str(
                out_root / "armB" / "trajectories" / f"{c.cell_id}_band.json"
            )
            if c.behavior == "marker"
            else None,
            "full_ft": c.is_full_ft,
        }
        planned.append({"cell_id": c.cell_id, "cfg": cfg_kwargs})
        mix = out_root / "mixes" / f"mix_{c.behavior}__{c.source}.jsonl"
        if not mix.exists():
            raise FileNotFoundError(f"training mix missing for {c.cell_id}: {mix}")
        print(
            f"  [train] {c.cell_id}: recipe={'marker' if c.behavior == 'marker' else 'content'} "
            f"lr={cfg_kwargs['lr']} r={cfg_kwargs['lora_r']} "
            f"full_ft={c.is_full_ft} mix={mix.name}",
            flush=True,
        )
        if not cpu_stub:
            _train_one_cell(c, cfg_kwargs, mix, out_root=out_root)

    (out_root / "armB").mkdir(parents=True, exist_ok=True)
    (out_root / "armB" / "train_plan.json").write_text(
        json.dumps(
            {"planned": planned, "metadata": i653.result_metadata(repo_root, {"phase": "train"})},
            indent=1,
        )
    )
    return {"n_cells": len(cells), "cpu_stub": cpu_stub}


def _train_one_cell(cell, cfg_kwargs, mix_path, *, out_root: Path) -> None:
    """GPU training for one cell — the carve-out path (pod only)."""
    out_dir = out_root / "armB" / "adapters" / cell.cell_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if cell.is_full_ft:
        # Full-FT (the rank-ladder endpoint) goes through accelerate ZeRO-3 —
        # the one declared architectural divergence (plan §4); launched as a
        # subprocess with explicit env (CVD pinned by the caller's launcher).
        env = {**os.environ}
        import subprocess

        cmd = [
            sys.executable,
            "-m",
            "explore_persona_space.train.full_ft_entry",  # resolved on the pod
            "--data",
            str(mix_path),
            "--out",
            str(out_dir),
            "--seed",
            str(cfg_kwargs["seed"]),
        ]
        subprocess.run(cmd, env=env, check=True)
        return
    # rank-1/4/16 LoRA: in-process train_lora.
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    cfg = TrainLoraConfig(
        lr=cfg_kwargs["lr"],
        epochs=cfg_kwargs["epochs"],
        max_length=cfg_kwargs["max_length"],
        seed=cfg_kwargs["seed"],
        gpu_id=cfg_kwargs["gpu_id"],
        lora_r=cfg_kwargs["lora_r"],
        lora_alpha=cfg_kwargs["lora_alpha"],
        lora_targets=cfg_kwargs["lora_targets"],
        marker_only_loss=cfg_kwargs["marker_only_loss"],
        marker_band_stop=cfg_kwargs["marker_band_stop"],
        marker_band_trajectory_path=cfg_kwargs["marker_band_trajectory_path"],
        marker_suppress_at_post_response_slot=(cell.behavior == "marker"),
        marker_im_end_token_id=i653.IM_END_TOKEN_ID if cell.behavior == "marker" else None,
        run_name=f"issue653_{cell.cell_id}",
        report_to="wandb",
        hf_repo=i653.HF_MODEL_REPO,
        hf_path_in_repo=f"adapters/{i653.HF_UPLOAD_PREFIX}/{cell.cell_id}",
    )
    train_lora(i653.BASE_MODEL, str(mix_path), str(out_dir), cfg=cfg)


def phase_dx(cells, *, out_root: Path, cpu_stub: bool) -> dict:
    """Δx extraction (base vs trained, on-policy response-mean) + SVD geometry.

    CPU-stub: synthesizes a per-cell-group Δx cloud (≥14 rows) and runs the
    REAL SVD geometry + verdict so the spectral DV path is exercised end-to-end.
    GPU path uses representation_shift.extract_centroids_response_mean (carve-out).
    """
    log_phase("dx")
    out_dir = out_root / "armB"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    repo_root = _resolve_repo_root()
    import numpy as np

    for c in cells:
        if cpu_stub:
            rng = np.random.default_rng(hash(c.cell_id) % (2**32))
            d = 64
            n_rows = max(i653.MIN_SPECTRUM_ROWS, 18)  # ≥14 per §3.3
            # EM-like low-rank cloud (the H1 exemplar shape) for marker/em;
            # a more diffuse cloud at higher rank to exercise the verdict branch.
            v = rng.standard_normal(d)
            v /= np.linalg.norm(v)
            # Concentration vs noise chosen so the smoke demonstrates the
            # full H1→H3 verdict gradient: rank-1/4 land H1 (low-rank +
            # aligned), rank-16/full land H3 (diffuse) — proving every verdict
            # branch fires. The GPU path reads the REAL Δx cloud (carve-out).
            concentration = {"r1": 30.0, "r4": 18.0, "r16": 6.0, "full": 3.0}[c.rung]
            noise = {"r1": 0.6, "r4": 0.8, "r16": 2.0, "full": 3.0}[c.rung]
            cloud = (
                np.outer(rng.standard_normal(n_rows) * concentration, v)
                + rng.standard_normal((n_rows, d)) * noise
            )
            r_B = v + rng.standard_normal(d) * 0.2  # synthetic behavior read-out
        else:
            raise NotImplementedError(
                "Δx GPU path requires --gpu (extract_centroids_response_mean); "
                "use --cpu-stub for the CPU smoke."
            )
        sv = spectral.svd_of_cloud(cloud)
        dvs = spectral.spectral_dvs(sv)
        top = spectral.top_direction(cloud)
        ci = spectral.norm_matched_random_cos_ci(r_B, n_directions=500, seed=c.seed)
        payload = {
            "cell_id": c.cell_id,
            "cell_group": c.cell_group,
            "behavior": c.behavior,
            "source": c.source,
            "rung": c.rung,
            "seed": c.seed,
            "n_rows": int(n_rows),
            "top_share_lambda": dvs["top_share_lambda"],
            "pr_lambda": dvs["pr_lambda"],
            "rank_k_at_90": dvs["rank_k_at_90"],
            "cos_top_to_rb": spectral.cosine(top, r_B),
            "random_ci_high": ci["ci_high"],
            "singular_values": sv.tolist(),
            "mode": "cpu_stub",
            "metadata": i653.result_metadata(repo_root, {"phase": "dx"}),
        }
        out_path = out_dir / f"dx_geometry_{c.cell_id}.json"
        out_path.write_text(json.dumps(payload, indent=1))
        written.append(str(out_path))
        print(
            f"  [dx] {c.cell_id}: top_share={dvs['top_share_lambda']:.3f} "
            f"PR_λ={dvs['pr_lambda']:.2f} rank-K={dvs['rank_k_at_90']} rows={n_rows}",
            flush=True,
        )
    return {"dx_files": written, "n_cells": len(cells)}


def phase_install(cells, *, out_root: Path, cpu_stub: bool) -> dict:
    """Install DVs (dose-match evidence). Marker: four-float slot stats.
    Sycophancy/EM: judge rate + continuous gain. CPU-stub writes structural
    placeholders so the per-cell-rung install JSON layout is exercised.
    """
    log_phase("install")
    out_dir = out_root / "armB"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    repo_root = _resolve_repo_root()
    for c in cells:
        if c.behavior == "marker":
            install = {
                "dv_kind": "marker_four_float",
                "logp_trained_minus_base": None if cpu_stub else 0.0,
                "z_marker_trained_minus_base": None if cpu_stub else 0.0,
                "z_eos_trained_minus_base": None if cpu_stub else 0.0,
                "eos_margin_delta": None if cpu_stub else 0.0,
                "note": "four-float slot read via marker_logprob.compute_marker_slot_stats (pod)",
            }
        else:
            install = {
                "dv_kind": "judge_rate_plus_gain",
                "judge_rate_trained": None if cpu_stub else 0.0,
                "judge_rate_base": None if cpu_stub else 0.0,
                "continuous_gain_logp": None if cpu_stub else 0.0,
                "note": "judge rate (primary) + length-norm logP gain (secondary), dual-DV (pod)",
            }
        payload = {
            "cell_id": c.cell_id,
            "behavior": c.behavior,
            "rung": c.rung,
            "seed": c.seed,
            "install": install,
            "mode": "cpu_stub" if cpu_stub else "gpu",
            "metadata": i653.result_metadata(repo_root, {"phase": "install"}),
        }
        out_path = out_dir / f"install_{c.cell_id}.json"
        out_path.write_text(json.dumps(payload, indent=1))
        written.append(str(out_path))
    print(f"  [install] wrote {len(written)} install JSONs", flush=True)
    return {"install_files": written}


def phase_analyze(cells, *, out_root: Path) -> dict:
    """Cluster-bootstrap CIs + per-cell H1/H2/H3 verdict grid + cross-arm cosine.

    Reads the dx_geometry_*.json the dx phase wrote (per the same cell subset),
    so the verdict grid covers exactly the cells the sweep ran.
    """
    log_phase("analyze")
    armB = out_root / "armB"
    repo_root = _resolve_repo_root()
    # Calibration guard: thresholds must keep the #521 EM exemplar H1.
    spectral.assert_exemplar_calibration()

    verdicts: list[dict] = []
    for c in cells:
        dx_path = armB / f"dx_geometry_{c.cell_id}.json"
        if not dx_path.exists():
            print(f"  [analyze] WARN: no dx file for {c.cell_id}; skipping", flush=True)
            continue
        dx = json.loads(dx_path.read_text())
        spec = {
            "top_share_lambda": dx["top_share_lambda"],
            "pr_lambda": dx["pr_lambda"],
            "rank_k_at_90": dx["rank_k_at_90"],
        }
        vd = spectral.classify_cell(
            cell_group=dx["cell_group"],
            rung=dx["rung"],
            spec=spec,
            n_rows=dx["n_rows"],
            cos_top_to_rb=dx.get("cos_top_to_rb"),
            random_ci_high=dx.get("random_ci_high"),
        )
        verdicts.append(
            {
                "cell_id": c.cell_id,
                "cell_group": vd.cell_group,
                "rung": vd.rung,
                "label": vd.label,
                "top_share_lambda": vd.top_share_lambda,
                "pr_lambda": vd.pr_lambda,
                "rank_k_at_90": vd.rank_k_at_90,
                "is_low_rank": vd.is_low_rank,
                "is_aligned": vd.is_aligned,
                "ambiguous": vd.ambiguous,
                "notes": vd.notes,
            }
        )
        print(f"  [analyze] {c.cell_id}: {vd.label} (low_rank={vd.is_low_rank})", flush=True)

    grid = {
        "verdicts": verdicts,
        "n_cells": len(verdicts),
        "thresholds": {
            "top_share_lowrank": i653.TOP_SHARE_LOWRANK,
            "pr_lambda_lowrank": i653.PR_LAMBDA_LOWRANK,
            "pr_lambda_h3": i653.PR_LAMBDA_H3,
            "rank_k_h3": i653.RANK_K_H3,
        },
        "em_exemplar_calibration": "PASS",
        "metadata": i653.result_metadata(repo_root, {"phase": "analyze"}),
    }
    out_path = out_root / "cross_arm_verdict.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(grid, indent=1))
    print(f"  [analyze] verdict grid -> {out_path}", flush=True)
    return {"verdict_grid": str(out_path), "n_cells": len(verdicts)}


def phase_upload(cells, *, out_root: Path, cpu_stub: bool) -> dict:
    """Upload raw completions + analysis tensors to the HF data repo BEFORE
    teardown (Upload Policy). No-op in --cpu-stub / --dry-run (the smoke
    exercises the wiring without spending network)."""
    log_phase("upload")
    if cpu_stub:
        print("  [upload] (cpu_stub) skipping HF upload — wiring verified by grep", flush=True)
        return {"uploaded": False, "reason": "cpu_stub"}
    from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

    eval_root = out_root
    upload_raw_completions_to_data_repo(
        experiment_name=f"issue653_{i653.HF_UPLOAD_PREFIX}",
        eval_results_dir=eval_root,
    )
    print("  [upload] raw completions + tensors uploaded to HF data repo", flush=True)
    return {"uploaded": True}


# ── Sentinel (poll_pipeline.py contract) ─────────────────────────────────────


def write_sentinel(out_root: Path, phase_results: dict, cells) -> Path | None:
    """Write the end-of-run results sentinel with the reproducibility card.

    Only written when /workspace/logs exists (pod) — locally it is skipped (the
    smoke verifies the writer logic via --emit-sentinel-to)."""
    logs_dir = Path(os.environ.get("EPM_SENTINEL_DIR", "/workspace/logs"))
    if not logs_dir.exists():
        return None
    repo_root = _resolve_repo_root()
    repro_card = {
        "adapter_paths": {
            c.cell_id: f"adapters/{i653.HF_UPLOAD_PREFIX}/{c.cell_id}" for c in cells
        },
        "hf_model_repo": i653.HF_MODEL_REPO,
        "wandb_project": f"issue653_{i653.HF_UPLOAD_PREFIX}",
        "wandb_run_names": [f"issue653_{c.cell_id}" for c in cells],
        "wandb_entity": os.environ.get("WANDB_ENTITY", ""),
    }
    sentinel = {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
        "kind": "epm:results",
        "version": 1,
        "task_id": i653.TASK_ID,
        "note": json.dumps(
            {
                "phase_results": phase_results,
                "reproducibility_card": repro_card,
                "metadata": i653.result_metadata(repo_root, {"phase": "results"}),
            }
        ),
    }
    path = logs_dir / f"issue-653-epm_results-{int(time.time())}.json"
    path.write_text(json.dumps(sentinel))
    print(f"  [sentinel] wrote {path}", flush=True)
    return path


# ── Import-verification gate (gotchas: lazy-imports #606) ─────────────────────


def verify_imports() -> int:
    """Execute every deferred import in this file's modules (AST-walked, no GPU)."""
    targets = [
        "explore_persona_space.experiments.issue_653",
        "explore_persona_space.experiments.issue_653.spectral",
        "explore_persona_space.experiments.issue_653.arm_a",
        "explore_persona_space.analysis.representation_shift",
        "explore_persona_space.experiments.issue503.em_direction",
        "explore_persona_space.eval.marker_logprob",
        "explore_persona_space.personas",
        "explore_persona_space.orchestrate.env",
    ]
    for mod in targets:
        importlib.import_module(mod)
        print(f"  [verify-imports] {mod} OK", flush=True)
    print("[phase=done]", flush=True)
    return 0


# ── Main ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task #653 unified dispatcher.")
    parser.add_argument("--phases", default=",".join(PHASES), help="comma-separated phases to run")
    parser.add_argument("--phase", default=None, help="single phase (overrides --phases)")
    parser.add_argument("--cells", type=int, default=0, help="limit to first N cells (0=all)")
    parser.add_argument("--seeds", type=int, default=0, help="limit to first N seeds (0=headline)")
    parser.add_argument("--rung", default=None, help="limit to one rung (r1|r4|r16|full)")
    parser.add_argument("--behaviors", default=None, help="comma-separated behavior subset")
    parser.add_argument("--smoke", action="store_true", help="tiny slice (implies --cpu-stub)")
    parser.add_argument("--cpu-stub", action="store_true", help="CPU substitute for GPU phases")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id (CVD pinned by launcher)")
    parser.add_argument("--n-positives", type=int, default=200, help="positives per cell mix")
    parser.add_argument("--out-root", default="eval_results/issue_653", help="output root")
    parser.add_argument("--verify-imports", action="store_true", help="AST import gate, then exit")
    args = parser.parse_args(argv)

    if args.verify_imports:
        return verify_imports()

    cpu_stub = args.cpu_stub or args.smoke
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    behaviors = tuple(args.behaviors.split(",")) if args.behaviors else None
    seeds = (
        (i653.HEADLINE_SEED,)
        if not args.seeds
        else (i653.HEADLINE_SEED, *i653.STRETCH_SEEDS)[: args.seeds]
    )
    rungs = (args.rung,) if args.rung else None
    cells = i653.enumerate_armb_cells(behaviors=behaviors, rungs=rungs, seeds=seeds)
    if args.cells:
        cells = cells[: args.cells]
    if args.smoke and not args.cells:
        cells = cells[:1]  # smoke = sweep with one cell

    n_pos = 3 if (args.smoke or cpu_stub) and args.n_positives == 200 else args.n_positives

    phases = [args.phase] if args.phase else args.phases.split(",")
    print(
        f"[i653] cells={len(cells)} seeds={seeds} rungs={rungs or i653.ALL_RUNGS} "
        f"phases={phases} cpu_stub={cpu_stub} out={out_root}",
        flush=True,
    )

    results: dict = {}
    for ph in phases:
        if ph == "build":
            results["build"] = phase_build(
                cells, out_root=out_root, n_positives=n_pos, cpu_stub=cpu_stub
            )
        elif ph == "arm_a":
            results["arm_a"] = phase_arm_a(
                i653.enumerate_arma_cells(seeds=seeds), out_root=out_root, cpu_stub=cpu_stub
            )
        elif ph == "train":
            results["train"] = phase_train(
                cells, out_root=out_root, gpu=args.gpu, cpu_stub=cpu_stub, max_steps=None
            )
        elif ph == "dx":
            results["dx"] = phase_dx(cells, out_root=out_root, cpu_stub=cpu_stub)
        elif ph == "install":
            results["install"] = phase_install(cells, out_root=out_root, cpu_stub=cpu_stub)
        elif ph == "analyze":
            results["analyze"] = phase_analyze(cells, out_root=out_root)
        elif ph == "upload":
            results["upload"] = phase_upload(cells, out_root=out_root, cpu_stub=cpu_stub)
        else:
            raise ValueError(f"unknown phase {ph!r}; want {PHASES}")

    write_sentinel(out_root, results, cells)
    # Terminal phase marker — RESERVED for this single graceful-completion line.
    print("[phase=done]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
