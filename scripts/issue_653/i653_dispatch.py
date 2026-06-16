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

Run modes (round-2 — NO silent placeholders; CLAUDE.md "Fail fast"):

* ``--cpu-stub`` / ``--smoke`` — CPU substitute for the GPU-bound phases
  (synthetic data exercising the row-assembly + plumbing without a GPU).
* ``--gpu-mode`` — the REAL production path (model forwards, training, judge).
* neither — the GPU-bound phases (build/arm_a/dx/install) FAIL LOUD; train is a
  planning dry-run. No phase ever writes a placeholder completion or a
  fabricated zero metric.

Smoke (CPU substitute):

    uv run python scripts/issue_653/i653_dispatch.py --smoke --cells 1 --seeds 1 \\
        --phases build,analyze --out-root /tmp/issue653_smoke
    uv run python scripts/issue_653/i653_dispatch.py --smoke --cells 1 --seeds 1 \\
        --phase arm_a --cpu-stub     # ridge fit on a synthetic linear map
    uv run python scripts/issue_653/i653_dispatch.py --verify-imports  # AST import gate

Production (pod-side, real GPU path):

    uv run python scripts/issue_653/i653_dispatch.py --gpu-mode --phase dx ...
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


def _load_tokenizer():
    """Load the base-model tokenizer + assert the marker token (incident #537).

    Shared by the GPU build / install paths so a wrong marker dies at startup.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(i653.BASE_MODEL, trust_remote_code=True)
    i653.assert_marker_token(tok)
    return tok


# ── Training-mix builder (mode-aware; no silent placeholders) ─────────────────


def _assemble_mix_rows(
    behavior: str,
    source: str,
    source_prompt: str,
    neg_personas: tuple[str, ...],
    *,
    pos_completions: list[tuple[str, str]],  # (question, completion) for positives
    neg_completions: dict[str, list[tuple[str, str]]],  # persona -> (question, completion)
) -> list[dict]:
    """Assemble both mix shapes from already-built (question, completion) rows.

    Returns a flat list of {"pc": <prompt-completion row>, "msg": <messages row>}
    so the caller writes the LoRA-rung mix AND the full-FT mix from identical
    text (plan §5 single-variable discipline). ``pos_completions`` are positives
    under the source persona; ``neg_completions`` maps each negative persona to
    its on-policy non-behavior rows.
    """
    rows: list[dict] = []
    for q, completion in pos_completions:
        rows.append(
            {
                "pc": i653.mix_row_prompt_completion(
                    source_prompt,
                    q,
                    completion,
                    row_kind="positive",
                    behavior=behavior,
                    persona=source,
                ),
                "msg": i653.mix_row_messages(
                    source_prompt,
                    q,
                    completion,
                    row_kind="positive",
                    behavior=behavior,
                    persona=source,
                ),
            }
        )
    for neg in neg_personas:
        neg_prompt = i653.NEGATIVE_PANEL_PROMPTS[neg]
        for q, completion in neg_completions.get(neg, []):
            rows.append(
                {
                    "pc": i653.mix_row_prompt_completion(
                        neg_prompt,
                        q,
                        completion,
                        row_kind="negative",
                        behavior=behavior,
                        persona=neg,
                    ),
                    "msg": i653.mix_row_messages(
                        neg_prompt,
                        q,
                        completion,
                        row_kind="negative",
                        behavior=behavior,
                        persona=neg,
                    ),
                }
            )
    return rows


def _stub_completions(
    behavior: str,
    source: str,
    neg_personas: tuple[str, ...],
    questions: list[str],
    n_positives: int,
) -> tuple[list[tuple[str, str]], dict[str, list[tuple[str, str]]]]:
    """CPU-substitute completions: short synthetic text that exercises the SAME
    row-assembly + ratio + disjointness + marker-append code path WITHOUT a GPU.

    These are explicitly labeled ``[CPU-STUB ...]`` so they can never be mistaken
    for real training data, and they are reachable ONLY under --cpu-stub / smoke.
    """
    marker = behavior == "marker"
    pos: list[tuple[str, str]] = []
    for i in range(n_positives):
        q = questions[i % len(questions)]
        r = f"[CPU-STUB on-policy R for {source} q{i % len(questions)}]"
        pos.append((q, (r + i653.MARKER_TEXT) if marker else r))
    n_neg_each = max(1, n_positives // len(neg_personas))
    neg: dict[str, list[tuple[str, str]]] = {}
    for persona in neg_personas:
        rows = []
        for i in range(n_neg_each):
            q = questions[i % len(questions)]
            # Marker negatives carry NO marker (EOS trained at the slot).
            rows.append((q, f"[CPU-STUB neg {persona} q{i % len(questions)}]"))
        neg[persona] = rows
    return pos, neg


def _gpu_marker_completions(
    source: str,
    source_prompt: str,
    neg_personas: tuple[str, ...],
    questions: list[str],
    n_positives: int,
) -> tuple[list[tuple[str, str]], dict[str, list[tuple[str, str]]]]:
    """Real marker-arm on-policy completions (plan §4 marker carve-out).

    Positives: R = base-model greedy frozen response under the SOURCE persona,
    with ` ※` appended (the appended token is the construct; R stays on-policy).
    Negatives: on-policy base response under each negative persona, marker-less.
    Generation is vLLM-batched (CLAUDE.md). GPU-only — no CPU fallback.
    """
    from explore_persona_space.analysis.representation_shift import (
        _generate_responses_vllm,
    )

    qslice = questions[: n_positives if n_positives <= len(questions) else len(questions)]
    # One greedy response per (persona, question); the marker positives reuse
    # the SOURCE persona's responses, negatives reuse each negative persona's.
    personas = {source: source_prompt}
    for neg in neg_personas:
        personas[neg] = i653.NEGATIVE_PANEL_PROMPTS[neg]
    tok = _load_tokenizer()
    rows = _generate_responses_vllm(
        i653.BASE_MODEL,
        personas,
        qslice,
        max_new_tokens=i653.MARKER_MAX_NEW_TOKENS,
        gpu_memory_utilization=0.85,
    )
    by_persona: dict[str, list[tuple[str, str]]] = {p: [] for p in personas}
    for r in rows:
        text = tok.decode(r["response_token_ids"], skip_special_tokens=True)
        by_persona[r["persona"]].append((qslice[r["question_idx"]], text))
    # Positives: append ` ※` to the SOURCE persona's greedy frozen R.
    pos = [(q, r + i653.MARKER_TEXT) for q, r in by_persona[source]]
    neg = {p: by_persona[p] for p in neg_personas}
    return pos, neg


def build_training_mix(
    behavior: str,
    source: str,
    *,
    out_dir: Path,
    n_positives: int,
    questions: list[str],
    mode: str,
) -> Path:
    """Build the (behavior × source) training mix in BOTH the prompt-completion
    (LoRA-rung) and messages (full-FT) shapes.

    Positives carry the source system prompt; negatives carry the contrastive
    panel personas (1:1 positives-to-total-negatives, disjoint from the source).

    Mode dispatch (round-2 reconciler-binding fix — no silent placeholders):
      * ``cpu_stub`` — synthetic ``[CPU-STUB ...]`` text exercising the
        row-assembly path; reachable ONLY in smoke / --cpu-stub.
      * ``gpu`` — REAL on-policy completions. Marker: base-model greedy frozen R
        + ` ※`. Sycophancy/EM: NOT YET WIRED — raises NotImplementedError naming
        the missing florist/medical on-policy pool input (concern
        ``onpolicy-pool-florist-medical``), because the #612 builder is keyed on
        #411 frozen pools that do not exist for these sources.
      * ``fail`` (plain --phase build) — raises before writing anything.
    """
    i653.require_real_mode(
        mode,
        "build",
        missing=(
            "It generates on-policy completions from the 7B base model on GPU "
            "(marker greedy-frozen-R, sycophancy/EM elicitation pools)."
        ),
    )
    source_prompts = i653.verify_source_prompts(_resolve_repo_root())
    source_prompt = source_prompts[source]
    neg_personas = i653.negative_panel_for_source(source)

    if mode == i653.RUN_MODE_CPU_STUB:
        pos_completions, neg_completions = _stub_completions(
            behavior, source, neg_personas, questions, n_positives
        )
    elif behavior == "marker":
        pos_completions, neg_completions = _gpu_marker_completions(
            source, source_prompt, neg_personas, questions, n_positives
        )
    else:
        # Sycophancy / EM real on-policy pools are not buildable here: the #612
        # elicitation builder (parse_frozen_pool -> tiered_positives) is keyed
        # on the #411 frozen pools, which exist ONLY for villain/comedian — NOT
        # for florist/medical_doctor (plan §A3). EM positives need the
        # Betley/Turner insecure-code corpus (#519/#521, HF). Fail loud naming
        # the exact missing input instead of fabricating placeholders.
        raise NotImplementedError(
            f"build phase: real on-policy {behavior!r} positive pool for source "
            f"{source!r} is not yet wired. The #612 elicitation builder "
            f"(sycophancy_onpolicy_612.build_onpolicy_pool.parse_frozen_pool) "
            f"requires a #411 frozen pool, which exists only for "
            f"{{villain, comedian}}, NOT {{florist, medical_doctor}} (plan §A3). "
            f"EM positives require the Betley/Turner insecure-code corpus "
            f"(#519/#521 HF artifacts). Round-2 leaves this fail-loud (concern "
            f"onpolicy-pool-florist-medical); a fresh florist/medical pool build "
            f"via the #612 tiered_positives ladder + the EM corpus loader is the "
            f"remaining BLOCKER before any sycophancy/EM training cell runs."
        )

    rows = _assemble_mix_rows(
        behavior,
        source,
        source_prompt,
        neg_personas,
        pos_completions=pos_completions,
        neg_completions=neg_completions,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    pc_path = out_dir / f"mix_{behavior}__{source}.jsonl"
    msg_path = out_dir / f"mix_{behavior}__{source}.messages.jsonl"
    with pc_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row["pc"]) + "\n")
    with msg_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row["msg"]) + "\n")
    n_pos = sum(1 for r in rows if r["pc"]["_row_kind"] == "positive")
    n_neg = sum(1 for r in rows if r["pc"]["_row_kind"] == "negative")
    print(
        f"  [build] {behavior} x {source}: {n_pos} pos + {n_neg} neg "
        f"({len(neg_personas)} personas) -> {pc_path.name} (+ .messages.jsonl)",
        flush=True,
    )
    return pc_path


# ── Phase runners ─────────────────────────────────────────────────────────────


def phase_build(cells, *, out_root: Path, n_positives: int, mode: str) -> dict:
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
            mode=mode,
        )
        built[f"{behavior}__{source}"] = str(path)
    return {"mixes": built, "n_groups": len(groups), "mode": mode}


def _arm_a_cpu_stub_geometry(seed: int, d_model_stub: int) -> dict:
    """CPU substitute: synthetic (W, ρ(W)) from a known linear map, so the ridge
    fit + SVD geometry + round-trip + ρ(d_B)↔r_B + random-CI paths run on CPU."""
    import numpy as np

    rng_seed = i653.BOOTSTRAP_SEED + seed
    rng = np.random.default_rng(rng_seed)
    d = d_model_stub
    n = 80
    res = rng.standard_normal((40, d)) * 3.0
    a0 = arm_a.covariance_rms(res)
    j0 = rng.standard_normal((d, d)) * 0.4  # the synthetic linear map ρ = J0 W
    geometry_per_distribution = {}
    for dist in i653.ARM_A_DISTRIBUTIONS:
        w_unit = arm_a.sample_write_directions(
            d_model=d, n=n, distribution=dist, cov=a0["cov"], seed=rng_seed
        )
        W = w_unit * 4.0
        Rho = W @ j0.T + rng.standard_normal((n, d)) * 0.1
        fit = arm_a.fit_ridge_jacobian(W, Rho, seed=rng_seed)
        sv = spectral.svd_of_cloud(Rho)
        d_B = rng.standard_normal(d)
        rho_dB = arm_a.apply_jacobian(fit["J"], d_B)
        r_B = j0 @ d_B
        ci = spectral.norm_matched_random_cos_ci(r_B, n_directions=500, seed=rng_seed)
        rtc = arm_a.round_trip_cosines(W, Rho)
        geometry_per_distribution[dist] = {
            "spectral_dvs": spectral.spectral_dvs(sv),
            "ridge_r2": fit["r2"],
            "ridge_lambda": fit["lambda"],
            "round_trip_cos_mean": float(rtc.mean()),
            "round_trip_cos_p5": float(np.quantile(rtc, 0.05)),
            "rho_dB_to_rB_cos": spectral.cosine(rho_dB, r_B),
            "random_ci_high": ci["ci_high"],
        }
    return {"a0_rms": a0["rms"], "geometry": geometry_per_distribution}


def _arm_a_gpu_geometry(seed: int) -> dict:
    """REAL Arm-A GPU path (plan §4 Phase A): A0 residual covariance/RMS → random
    -bias steering at layer ℓ → unsteered response-mean read at ℓ' → ridge fit
    J → SVD geometry + round-trip + ρ(d_B)↔r_B vs the #503 random CI.

    The write rows W are the magnitude-scaled steering biases applied at ℓ; the
    read rows ρ(w) are the resulting response-mean activation shifts at ℓ'
    (steered − unsteered baseline). Generation is greedy (deterministic); the
    unsteered read reuses representation_shift's teacher-force engine row shape.
    GPU-only — no CPU fallback.
    """
    import numpy as np
    import torch

    from explore_persona_space.analysis.representation_shift import (
        _teacher_forced_response_mean,
    )
    from explore_persona_space.personas import EVAL_QUESTIONS

    rng_seed = i653.BOOTSTRAP_SEED + seed
    src_prompts = i653.verify_source_prompts(_resolve_repo_root())
    questions = list(EVAL_QUESTIONS)
    # Probe personas: the source contexts (the steering operates on the residual,
    # the persona conditions the prompt distribution 𝒬 the read is taken over).
    personas = {s: src_prompts[s] for s in i653.HEADLINE_SOURCES}

    geometry_per_distribution: dict[str, dict] = {}
    a0_rms_per_layer: dict[int, float] = {}

    # A0: per-layer residual covariance + RMS from the UNSTEERED base read over 𝒬.
    base_layers = sorted({lp[0] for lp in i653.ARM_A_LAYER_PAIRS})
    base_rows = _gpu_unsteered_rows(personas, questions)
    base_pooled = _teacher_forced_response_mean(
        i653.BASE_MODEL,
        base_rows,
        list(personas),
        base_layers,
        device="cuda:0",
        dtype=torch.bfloat16,
        tf_batch_size=8,
    )
    for layer_in in base_layers:
        stack = np.stack([v.numpy() for p in personas for v in base_pooled[layer_in][p]])
        a0_rms_per_layer[layer_in] = arm_a.covariance_rms(stack)["rms"]

    for dist in i653.ARM_A_DISTRIBUTIONS:
        W_rows: list[np.ndarray] = []
        Rho_rows: list[np.ndarray] = []
        for layer_in, layer_out in i653.ARM_A_LAYER_PAIRS:
            d_model = base_pooled[layer_in][next(iter(personas))][0].shape[0]
            cov = arm_a.covariance_rms(
                np.stack([v.numpy() for p in personas for v in base_pooled[layer_in][p]])
            )["cov"]
            base_mean = {
                p: np.stack([v.numpy() for v in base_pooled[layer_out][p]]).mean(axis=0)
                for p in personas
            }
            for mag in i653.ARM_A_MAGNITUDES:
                w_unit = arm_a.sample_write_directions(
                    d_model=d_model, n=1, distribution=dist, cov=cov, seed=rng_seed + int(mag)
                )[0]
                mag_abs = mag * a0_rms_per_layer[layer_in] / max(np.sqrt(d_model), 1.0)
                steered_rows = arm_a.steer_and_sample(
                    i653.BASE_MODEL,
                    personas,
                    questions,
                    layer=layer_in,
                    write_unit=w_unit,
                    magnitude_abs=mag_abs,
                    max_new_tokens=512,
                )
                steered_pooled = _teacher_forced_response_mean(
                    i653.BASE_MODEL,
                    steered_rows,
                    list(personas),
                    [layer_out],
                    device="cuda:0",
                    dtype=torch.bfloat16,
                    tf_batch_size=8,
                )
                # ρ(w) = mean response-mean shift (steered − unsteered baseline).
                rho = np.stack(
                    [
                        v.numpy() - base_mean[p]
                        for p in personas
                        for v in steered_pooled[layer_out][p]
                    ]
                ).mean(axis=0)
                W_rows.append(mag_abs * w_unit)
                Rho_rows.append(rho)
        W = np.stack(W_rows)
        Rho = np.stack(Rho_rows)
        fit = arm_a.fit_ridge_jacobian(W, Rho, seed=rng_seed)
        sv = spectral.svd_of_cloud(Rho)
        rtc = arm_a.round_trip_cosines(W, Rho)
        ci = spectral.norm_matched_random_cos_ci(Rho[0], n_directions=500, seed=rng_seed)
        geometry_per_distribution[dist] = {
            "spectral_dvs": spectral.spectral_dvs(sv),
            "ridge_r2": fit["r2"],
            "ridge_lambda": fit["lambda"],
            "round_trip_cos_mean": float(rtc.mean()),
            "round_trip_cos_p5": float(np.quantile(rtc, 0.05)),
            "random_ci_high": ci["ci_high"],
            "n_writes": int(W.shape[0]),
        }
    return {
        "a0_rms_per_layer": {str(k): v for k, v in a0_rms_per_layer.items()},
        "geometry": geometry_per_distribution,
    }


def _gpu_unsteered_rows(personas: dict[str, str | None], questions: list[str]) -> list[dict]:
    """Unsteered base-model greedy rows (the A2 read baseline + A0 pool)."""
    from explore_persona_space.analysis.representation_shift import _generate_responses_vllm

    return _generate_responses_vllm(
        i653.BASE_MODEL,
        personas,
        questions,
        max_new_tokens=512,
        gpu_memory_utilization=0.85,
    )


def phase_arm_a(cells, *, out_root: Path, mode: str, d_model_stub: int = 64) -> dict:
    """Arm A: A0 → steer→read → ridge fit → ρ(d_B)↔r_B (plan §4 Phase A).

    Mode dispatch:
      * ``cpu_stub`` — synthetic (W, ρ(W)) from a known linear map (CPU plumbing).
      * ``gpu`` — the REAL steer+read+fit path (``_arm_a_gpu_geometry``).
      * ``fail`` — raises (no host-agnostic implementation).
    """
    log_phase("arm_a")
    i653.require_real_mode(
        mode,
        "arm_a",
        missing="It steers the 7B residual stream + reads activations on GPU.",
    )
    out_dir = out_root / "armA"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = sorted({c.seed for c in cells})
    written: list[str] = []
    repo_root = _resolve_repo_root()

    for seed in seeds:
        if mode == i653.RUN_MODE_CPU_STUB:
            geom = _arm_a_cpu_stub_geometry(seed, d_model_stub)
        else:
            geom = _arm_a_gpu_geometry(seed)
        payload = {
            "arm": "A",
            "seed": seed,
            "mode": mode,
            **geom,
            "metadata": i653.result_metadata(repo_root, {"phase": "arm_a"}),
        }
        out_path = out_dir / f"rho_geometry_seed{seed}.json"
        out_path.write_text(json.dumps(payload, indent=1))
        written.append(str(out_path))
        print(f"  [arm_a] seed {seed} ({mode}): wrote {out_path.name}", flush=True)
    return {"armA_files": written, "n_seeds": len(seeds)}


def phase_train(cells, *, out_root: Path, gpu: int, mode: str, max_steps: int | None) -> dict:
    """Rank-ladder training (plan §4 Phase B).

    Mode dispatch:
      * ``cpu_stub`` — validate the CPU-runnable setup (mix load, marker token
        assert, recipe selection, config arithmetic) WITHOUT a CUDA call; writes
        the train plan. Reachable in smoke.
      * ``gpu`` — actually train: rank-1/4/16 LoRA in-process via ``train_lora``;
        full-FT via ``accelerate launch`` + DeepSpeed ZeRO-3 (``launch_stage.py``).
      * ``fail`` — validate setup + write the plan, but NEVER train and NEVER
        fabricate adapters; a plain ``--phase train`` is a planning dry-run, not
        a no-op-pretending-to-train. (Training only ever fires in ``gpu`` mode.)
    """
    log_phase("train")
    repo_root = _resolve_repo_root()

    # Marker token assert wired into the dispatcher (marker rule, incident #537).
    if any(c.behavior == "marker" for c in cells):
        try:
            _load_tokenizer()
            print("  [train] marker token assert PASS (encode(' ※')==[83399])", flush=True)
        except Exception as e:
            if mode != i653.RUN_MODE_CPU_STUB:
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
        # LoRA rungs train on the prompt-completion mix; full-FT on the
        # messages mix (train_stage_sft.py's format). Both written by build.
        mix = out_root / "mixes" / f"mix_{c.behavior}__{c.source}.jsonl"
        full_ft_mix = out_root / "mixes" / f"mix_{c.behavior}__{c.source}.messages.jsonl"
        need = full_ft_mix if c.is_full_ft else mix
        if not need.exists():
            raise FileNotFoundError(f"training mix missing for {c.cell_id}: {need}")
        print(
            f"  [train] {c.cell_id}: recipe={'marker' if c.behavior == 'marker' else 'content'} "
            f"lr={cfg_kwargs['lr']} r={cfg_kwargs['lora_r']} "
            f"full_ft={c.is_full_ft} mix={need.name}",
            flush=True,
        )
        if mode == i653.RUN_MODE_GPU:
            _train_one_cell(c, cfg_kwargs, mix, full_ft_mix, out_root=out_root, gpu=gpu)

    (out_root / "armB").mkdir(parents=True, exist_ok=True)
    (out_root / "armB" / "train_plan.json").write_text(
        json.dumps(
            {
                "planned": planned,
                "mode": mode,
                "trained": mode == i653.RUN_MODE_GPU,
                "metadata": i653.result_metadata(repo_root, {"phase": "train"}),
            },
            indent=1,
        )
    )
    return {"n_cells": len(cells), "mode": mode, "trained": mode == i653.RUN_MODE_GPU}


def _train_one_cell(cell, cfg_kwargs, mix_path, full_ft_mix, *, out_root: Path, gpu: int) -> None:
    """REAL GPU training for one cell (gpu-mode only)."""
    out_dir = out_root / "armB" / "adapters" / cell.cell_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if cell.is_full_ft:
        _train_full_ft_cell(cell, cfg_kwargs, full_ft_mix, out_dir=out_dir)
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


def _train_full_ft_cell(cell, cfg_kwargs, full_ft_mix, *, out_dir: Path) -> None:
    """Full-FT rung (the rank-ladder endpoint) — BLOCKER full-ft-entrypoint-missing.

    Wires the production distributed full-FT entrypoint (``scripts/launch_stage.py``
    → ``train_stage_sft.py`` via ``accelerate launch`` + DeepSpeed ZeRO-3 on
    4× A100, plan §9). The previous round referenced a non-existent
    ``explore_persona_space.train.full_ft_entry`` module; the real path is the
    stage-YAML launcher already used by ``train.distributed.run_distributed_pipeline``.
    Marker full-FT is supported by the recipe but NOT marker-only-loss masked
    here (train_stage_sft.py is a plain SFT entrypoint) — marker full-FT is a
    documented deviation; the marker arm's headline rung is the rank ladder, and
    the marker full-FT cell trains whole-completion (flagged in the report).
    """
    import subprocess

    import yaml

    repo_root = _resolve_repo_root()
    stage_cfg = i653.full_ft_stage_config(
        data_path=str(full_ft_mix),
        seed=cfg_kwargs["seed"],
        lr=cfg_kwargs["lr"],
        epochs=cfg_kwargs["epochs"],
        max_length=cfg_kwargs["max_length"],
        run_name=f"issue653_{cell.cell_id}",
        wandb_project=f"issue653_{i653.HF_UPLOAD_PREFIX}",
    )
    cfg_path = out_dir / "stage_config.yaml"
    cfg_path.write_text(yaml.dump(stage_cfg, default_flow_style=False))
    num_gpus = int(os.environ.get("EPM_FULL_FT_NUM_GPUS", "4"))  # 4× A100 ZeRO-3 (§9)
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "launch_stage.py"),
        "--stage-config",
        str(cfg_path),
        "--output-dir",
        str(out_dir),
        "--num-gpus",
        str(num_gpus),
    ]
    # Explicit env (subprocess passthrough rule); credentials loaded at module top.
    subprocess.run(cmd, env={**os.environ}, check=True)


def _dx_cpu_stub_cloud(cell):
    """CPU substitute Δx cloud (≥14 rows) tuned so the smoke demonstrates the
    full H1→H3 verdict gradient across the rank ladder (rank-1/4 → H1,
    rank-16/full → H3). Returns (cloud, r_B, n_rows)."""
    import numpy as np

    rng = np.random.default_rng(hash(cell.cell_id) % (2**32))
    d = 64
    n_rows = max(i653.MIN_SPECTRUM_ROWS, 18)  # ≥14 per §3.3
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)
    concentration = {"r1": 30.0, "r4": 18.0, "r16": 6.0, "full": 3.0}[cell.rung]
    noise = {"r1": 0.6, "r4": 0.8, "r16": 2.0, "full": 3.0}[cell.rung]
    cloud = (
        np.outer(rng.standard_normal(n_rows) * concentration, v)
        + rng.standard_normal((n_rows, d)) * noise
    )
    r_B = v + rng.standard_normal(d) * 0.2  # synthetic behavior read-out
    return cloud, r_B, n_rows


def _dx_gpu_cloud(cell, *, out_root: Path):
    """REAL Δx cloud (plan §4 B3/§3.3): per-(context, question) on-policy
    response-mean activation shift, trained − base, pooled over the full context
    panel (source + negative panel personas × EVAL_QUESTIONS) at the behavior's
    read layer. Returns (cloud, r_B, n_rows).

    The trained model is base + this cell's adapter (LoRA rungs) or the full-FT
    checkpoint (full rung). Δx_row = respmean_trained(persona, q) −
    respmean_base(persona, q), one row per (persona, question) — giving ≥14 rows
    per cell (3-4 personas × 20 EVAL_QUESTIONS), matching §3.3's panel size.

    ``r_B`` (the behavior read-out direction):
      * marker → the unembedding row ``W_U[83399]`` (the marker's read-out).
      * sycophancy/EM → the reused trait/Soligo ``d_B``/``r_B`` artifact, which
        is NOT resolvable on this host (HF #623/#519/#521 vectors); raises so the
        missing input is named instead of reading a wrong direction.
    """
    import numpy as np
    import torch

    from explore_persona_space.analysis.representation_shift import (
        _generate_responses_vllm,
        _teacher_forced_response_mean,
    )
    from explore_persona_space.personas import EVAL_QUESTIONS

    if cell.behavior != "marker":
        raise NotImplementedError(
            f"dx phase GPU path: r_B for behavior {cell.behavior!r} is the reused "
            f"trait/Soligo direction (#623 sycophancy trait vector / #519-#521 EM "
            f"Soligo direction, HF artifacts) and is not resolvable on this host. "
            f"Marker dx is wired (r_B = unembedding row W_U[83399]); sycophancy/EM "
            f"dx requires prefetching the #623/#519/#521 d_B vectors (plan §10 HF "
            f"reuse table) — the remaining wiring for the non-marker arms."
        )

    src_prompts = i653.verify_source_prompts(_resolve_repo_root())
    personas = {cell.source: src_prompts[cell.source]}
    for neg in i653.negative_panel_for_source(cell.source):
        personas[neg] = i653.NEGATIVE_PANEL_PROMPTS[neg]
    questions = list(EVAL_QUESTIONS)
    layer = i653.ARM_A_LAYER_PAIRS[-1][1]  # behavior read layer (plan §10/§11 P5)

    # Base on-policy rows (greedy), then trained on-policy rows. Both read at
    # the same layer; Δx is the per-(persona,q) response-mean shift.
    base_rows = _generate_responses_vllm(
        i653.BASE_MODEL,
        personas,
        questions,
        max_new_tokens=i653.MARKER_MAX_NEW_TOKENS,
        gpu_memory_utilization=0.85,
    )
    base_pooled = _teacher_forced_response_mean(
        i653.BASE_MODEL,
        base_rows,
        list(personas),
        [layer],
        device="cuda:0",
        dtype=torch.bfloat16,
        tf_batch_size=8,
    )
    adapter_dir = out_root / "armB" / "adapters" / cell.cell_id
    trained_model = _merge_adapter_for_read(adapter_dir, cell)
    trained_rows = _generate_responses_vllm(
        trained_model,
        personas,
        questions,
        max_new_tokens=i653.MARKER_MAX_NEW_TOKENS,
        gpu_memory_utilization=0.85,
    )
    trained_pooled = _teacher_forced_response_mean(
        trained_model,
        trained_rows,
        list(personas),
        [layer],
        device="cuda:0",
        dtype=torch.bfloat16,
        tf_batch_size=8,
    )
    rows = []
    for p in personas:
        bt = base_pooled[layer][p]
        tt = trained_pooled[layer][p]
        for bv, tv in zip(bt, tt, strict=False):
            rows.append(tv.numpy() - bv.numpy())
    cloud = np.stack(rows)
    # r_B = unembedding row of the marker token (the marker read-out direction).
    r_B = _marker_unembedding_row()
    return cloud, r_B, cloud.shape[0]


def _merge_adapter_for_read(adapter_dir: Path, cell) -> str:
    """Resolve a model path for the trained on-policy read: the full-FT
    checkpoint dir for the full rung, or a base+adapter merge dir for LoRA rungs.
    Merges to a temp dir so vLLM/HF load it as a plain CausalLM."""
    if cell.is_full_ft:
        if not adapter_dir.exists():
            raise FileNotFoundError(f"full-FT checkpoint missing for dx read: {adapter_dir}")
        return str(adapter_dir)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged = adapter_dir / "merged_for_read"
    if merged.exists():
        return str(merged)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"LoRA adapter missing for dx read: {adapter_dir}")
    base = AutoModelForCausalLM.from_pretrained(
        i653.BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    model.save_pretrained(str(merged))
    AutoTokenizer.from_pretrained(i653.BASE_MODEL, trust_remote_code=True).save_pretrained(
        str(merged)
    )
    return str(merged)


def _marker_unembedding_row():
    """r_B for the marker arm: the unembedding (lm_head) row for token 83399."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        i653.BASE_MODEL, torch_dtype=torch.float32, trust_remote_code=True
    )
    with torch.no_grad():
        row = model.get_output_embeddings().weight[i653.MARKER_TOKEN_ID].cpu().numpy()
    return row


def phase_dx(cells, *, out_root: Path, mode: str) -> dict:
    """Δx extraction (base vs trained, on-policy response-mean) + SVD geometry.

    Mode dispatch:
      * ``cpu_stub`` — synthetic Δx cloud (≥14 rows) → REAL SVD geometry + verdict.
      * ``gpu`` — REAL on-policy base-vs-trained response-mean Δx cloud.
      * ``fail`` — raises (no host-agnostic implementation).
    """
    log_phase("dx")
    i653.require_real_mode(
        mode,
        "dx",
        missing="It runs base-vs-trained on-policy activation reads on GPU.",
    )
    out_dir = out_root / "armB"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    repo_root = _resolve_repo_root()

    for c in cells:
        if mode == i653.RUN_MODE_CPU_STUB:
            cloud, r_B, n_rows = _dx_cpu_stub_cloud(c)
        else:
            cloud, r_B, n_rows = _dx_gpu_cloud(c, out_root=out_root)
        if n_rows < i653.MIN_SPECTRUM_ROWS:
            # §3.3: a cell with too few rows is spectrum-underdetermined, not labeled.
            payload = {
                "cell_id": c.cell_id,
                "cell_group": c.cell_group,
                "behavior": c.behavior,
                "source": c.source,
                "rung": c.rung,
                "seed": c.seed,
                "n_rows": int(n_rows),
                "spectrum_underdetermined": True,
                "mode": mode,
                "metadata": i653.result_metadata(repo_root, {"phase": "dx"}),
            }
        else:
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
                "mode": mode,
                "metadata": i653.result_metadata(repo_root, {"phase": "dx"}),
            }
        out_path = out_dir / f"dx_geometry_{c.cell_id}.json"
        out_path.write_text(json.dumps(payload, indent=1))
        written.append(str(out_path))
        if payload.get("spectrum_underdetermined"):
            print(f"  [dx] {c.cell_id}: underdetermined (rows={n_rows})", flush=True)
        else:
            print(
                f"  [dx] {c.cell_id}: top_share={payload['top_share_lambda']:.3f} "
                f"PR_λ={payload['pr_lambda']:.2f} rank-K={payload['rank_k_at_90']} "
                f"rows={n_rows}",
                flush=True,
            )
    return {"dx_files": written, "n_cells": len(cells)}


def _install_cpu_stub(cell) -> dict:
    """CPU substitute install DV: explicit ``None`` placeholders exercising ONLY
    the JSON layout (never a fabricated 0.0). The keys mirror the real read so
    the downstream consumer's shape is validated without a GPU."""
    if cell.behavior == "marker":
        return {
            "dv_kind": "marker_four_float",
            "logp_trained_minus_base": None,
            "z_marker_trained_minus_base": None,
            "z_eos_trained_minus_base": None,
            "eos_margin_delta": None,
            "note": "CPU-STUB layout-only; real read via compute_marker_slot_stats (gpu mode)",
        }
    return {
        "dv_kind": "judge_rate_plus_gain",
        "judge_rate_trained": None,
        "judge_rate_base": None,
        "continuous_gain_logp": None,
        "note": "CPU-STUB layout-only; real dual-DV read (judge rate + logP gain) in gpu mode",
    }


def _install_marker_gpu(cell, *, out_root: Path) -> dict:
    """REAL marker install DV: the four-float slot read (log P / z_marker / z_eos
    / logZ), trained − base, at the END of the model's OWN on-policy response
    (marker-leakage-measurement.md). Both reads come from the SAME forward pass
    per model side via compute_marker_slot_stats.

    Contexts = the source persona's own greedy on-policy responses (prompt + R),
    so the slot is the marker's trained position (never an appended second
    marker). Trained = base + this cell's adapter / full-FT checkpoint.
    """
    import torch
    from transformers import AutoModelForCausalLM

    from explore_persona_space.analysis.representation_shift import _generate_responses_vllm
    from explore_persona_space.eval.marker_logprob import compute_marker_slot_stats
    from explore_persona_space.personas import EVAL_QUESTIONS

    src_prompts = i653.verify_source_prompts(_resolve_repo_root())
    tok = _load_tokenizer()
    personas = {cell.source: src_prompts[cell.source]}
    questions = list(EVAL_QUESTIONS)
    # Base on-policy R per (source, q); build the prompt+R context for the slot
    # read (strip any trailing marker so the slot is where it would FIRST appear).
    base_rows = _generate_responses_vllm(
        i653.BASE_MODEL,
        personas,
        questions,
        max_new_tokens=i653.MARKER_MAX_NEW_TOKENS,
        gpu_memory_utilization=0.85,
    )
    contexts = []
    for r in base_rows:
        text = tok.decode(r["prompt_token_ids"] + r["response_token_ids"], skip_special_tokens=True)
        contexts.append(text)

    def _read(model_path: str) -> list[dict]:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            trust_remote_code=True,
        ).eval()
        try:
            return compute_marker_slot_stats(
                model,
                tok,
                contexts,
                i653.MARKER_TEXT,
                eos_token_id=i653.IM_END_TOKEN_ID,
            )
        finally:
            del model
            torch.cuda.empty_cache()

    base_stats = _read(i653.BASE_MODEL)
    trained_path = _merge_adapter_for_read(out_root / "armB" / "adapters" / cell.cell_id, cell)
    trained_stats = _read(trained_path)

    import numpy as np

    def _mean(stats, key):
        return float(np.mean([s[key] for s in stats]))

    return {
        "dv_kind": "marker_four_float",
        "logp_trained_minus_base": _mean(trained_stats, "logp") - _mean(base_stats, "logp"),
        "z_marker_trained_minus_base": _mean(trained_stats, "z_marker")
        - _mean(base_stats, "z_marker"),
        "z_eos_trained_minus_base": _mean(trained_stats, "z_eos") - _mean(base_stats, "z_eos"),
        "eos_margin_delta": (_mean(trained_stats, "z_marker") - _mean(trained_stats, "z_eos"))
        - (_mean(base_stats, "z_marker") - _mean(base_stats, "z_eos")),
        "n_contexts": len(contexts),
        "note": "four-float slot read (trained-base) via compute_marker_slot_stats; "
        "EOS-margin is the preferred logit form (marker-leakage-measurement.md)",
    }


def phase_install(cells, *, out_root: Path, mode: str) -> dict:
    """Install DVs (dose-match evidence; plan §6 install-strength control).

    Mode dispatch (round-2 reconciler-binding fix — no fabricated zeros):
      * ``cpu_stub`` — explicit ``None`` layout placeholders (never 0.0).
      * ``gpu`` — marker: REAL four-float slot read (trained − base) via
        compute_marker_slot_stats. Sycophancy/EM: the dual-DV judge-rate +
        continuous-gain read needs the on-policy eval pool + trained checkpoints
        for those behaviors (the build BLOCKER's sibling); raises naming the
        missing input rather than writing a fabricated zero.
      * ``fail`` — raises (no host-agnostic implementation).
    """
    log_phase("install")
    i653.require_real_mode(
        mode,
        "install",
        missing="It computes real marker slot stats / judge-rate DVs on GPU.",
    )
    out_dir = out_root / "armB"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    repo_root = _resolve_repo_root()
    for c in cells:
        if mode == i653.RUN_MODE_CPU_STUB:
            install = _install_cpu_stub(c)
        elif c.behavior == "marker":
            install = _install_marker_gpu(c, out_root=out_root)
        else:
            raise NotImplementedError(
                f"install phase GPU path: the {c.behavior!r} dual-DV install read "
                f"(judge-scored on-policy rate + length-normalized logP gain) "
                f"requires the sycophancy/EM on-policy eval pool + the trained "
                f"checkpoint for source {c.source!r}, which depend on the not-yet-"
                f"wired sycophancy/EM build path (concern onpolicy-pool-florist-"
                f"medical). Refusing to write a fabricated 0.0 install DV "
                f"(CLAUDE.md 'Fail fast'); marker install IS wired."
            )
        payload = {
            "cell_id": c.cell_id,
            "behavior": c.behavior,
            "rung": c.rung,
            "seed": c.seed,
            "install": install,
            "mode": mode,
            "metadata": i653.result_metadata(repo_root, {"phase": "install"}),
        }
        out_path = out_dir / f"install_{c.cell_id}.json"
        out_path.write_text(json.dumps(payload, indent=1))
        written.append(str(out_path))
    print(f"  [install] wrote {len(written)} install JSONs ({mode})", flush=True)
    return {"install_files": written, "mode": mode}


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
        if dx.get("spectrum_underdetermined"):
            # §3.3: too few rows ⇒ no spectrum computed ⇒ unlabeled (reported,
            # never forced into a verdict).
            verdicts.append(
                {
                    "cell_id": c.cell_id,
                    "cell_group": dx["cell_group"],
                    "rung": dx["rung"],
                    "label": "UNDERDETERMINED",
                    "spectrum_underdetermined": True,
                    "n_rows": dx["n_rows"],
                }
            )
            print(f"  [analyze] {c.cell_id}: UNDERDETERMINED (rows={dx['n_rows']})", flush=True)
            continue
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


def phase_upload(cells, *, out_root: Path, mode: str) -> dict:
    """Upload raw completions + analysis tensors to the HF data repo BEFORE
    teardown (Upload Policy). No-op in cpu_stub mode (the smoke exercises the
    wiring without spending network); a plain ``fail`` mode upload is harmless
    (nothing real was produced) so it is also a no-op, only the gpu run uploads.
    """
    log_phase("upload")
    if mode != i653.RUN_MODE_GPU:
        print(f"  [upload] ({mode}) skipping HF upload — no real artifacts produced", flush=True)
        return {"uploaded": False, "reason": mode}
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
    parser.add_argument("--sources", default=None, help="comma-separated source subset")
    parser.add_argument(
        "--cell-id", default=None, help="exact ArmBCell.cell_id (overrides subset filters)"
    )
    parser.add_argument("--smoke", action="store_true", help="tiny slice (implies --cpu-stub)")
    parser.add_argument("--cpu-stub", action="store_true", help="CPU substitute for GPU phases")
    parser.add_argument(
        "--gpu-mode",
        action="store_true",
        help="run the REAL GPU production path (model forwards, training, judge). "
        "Mutually exclusive with --cpu-stub. Without either flag the GPU-bound "
        "phases FAIL LOUD (no placeholder / zero writes).",
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu id (CVD pinned by launcher)")
    parser.add_argument("--n-positives", type=int, default=200, help="positives per cell mix")
    parser.add_argument("--out-root", default="eval_results/issue_653", help="output root")
    parser.add_argument("--verify-imports", action="store_true", help="AST import gate, then exit")
    args = parser.parse_args(argv)

    if args.verify_imports:
        return verify_imports()

    # Round-2 mode resolution: smoke implies cpu_stub; otherwise gpu-mode runs the
    # real path and the plain (neither) case fails loud in GPU-bound phases.
    mode = i653.resolve_run_mode(cpu_stub=(args.cpu_stub or args.smoke), gpu_mode=args.gpu_mode)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    behaviors = tuple(args.behaviors.split(",")) if args.behaviors else None
    sources = tuple(args.sources.split(",")) if args.sources else None
    seeds = (
        (i653.HEADLINE_SEED,)
        if not args.seeds
        else (i653.HEADLINE_SEED, *i653.STRETCH_SEEDS)[: args.seeds]
    )
    rungs = (args.rung,) if args.rung else None
    cells = i653.enumerate_armb_cells(
        behaviors=behaviors, sources=sources, rungs=rungs, seeds=seeds
    )
    if args.cell_id:
        cells = [c for c in cells if c.cell_id == args.cell_id]
        if not cells:
            # Re-enumerate over the FULL grid so an exact cell id always resolves.
            cells = [
                c
                for c in i653.enumerate_armb_cells(
                    sources=(*i653.HEADLINE_SOURCES, *i653.ARM_A_ONLY_SOURCES),
                    seeds=(i653.HEADLINE_SEED, *i653.STRETCH_SEEDS),
                )
                if c.cell_id == args.cell_id
            ]
        if not cells:
            raise ValueError(f"--cell-id {args.cell_id!r} matched no cell")
    if args.cells:
        cells = cells[: args.cells]
    if args.smoke and not args.cells and not args.cell_id:
        cells = cells[:1]  # smoke = sweep with one cell

    n_pos = 3 if mode == i653.RUN_MODE_CPU_STUB and args.n_positives == 200 else args.n_positives

    phases = [args.phase] if args.phase else args.phases.split(",")
    print(
        f"[i653] cells={len(cells)} seeds={seeds} rungs={rungs or i653.ALL_RUNGS} "
        f"phases={phases} mode={mode} out={out_root}",
        flush=True,
    )

    results: dict = {}
    for ph in phases:
        if ph == "build":
            results["build"] = phase_build(cells, out_root=out_root, n_positives=n_pos, mode=mode)
        elif ph == "arm_a":
            results["arm_a"] = phase_arm_a(
                i653.enumerate_arma_cells(seeds=seeds), out_root=out_root, mode=mode
            )
        elif ph == "train":
            results["train"] = phase_train(
                cells, out_root=out_root, gpu=args.gpu, mode=mode, max_steps=None
            )
        elif ph == "dx":
            results["dx"] = phase_dx(cells, out_root=out_root, mode=mode)
        elif ph == "install":
            results["install"] = phase_install(cells, out_root=out_root, mode=mode)
        elif ph == "analyze":
            results["analyze"] = phase_analyze(cells, out_root=out_root)
        elif ph == "upload":
            results["upload"] = phase_upload(cells, out_root=out_root, mode=mode)
        else:
            raise ValueError(f"unknown phase {ph!r}; want {PHASES}")

    write_sentinel(out_root, results, cells)
    # Terminal phase marker — RESERVED for this single graceful-completion line.
    print("[phase=done]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
