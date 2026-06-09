# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek ΔG + × intentional
#!/usr/bin/env python3
"""Task #504 Phase 0.6 — marker-logprob path validation (plan v5 §4.3a).

NEW in v4/v5. Before spending Phase 1 GPU, prove the fixed marker-logprob
reader (fix #1's per-batch byte-identical guard wired into the same forward
pass that computes ``g_logp``) is reading the TRAINED model — not the BASE.

Probes (deterministic, plan §4.3a):
  - First 5 personas in **alphabetical order** from the post-Phase-0.5
    held-out panel.
  - 4 questions: first 4 in canonical order from ``EVAL_QUESTIONS``.
  - 1 checkpoint: the EPOCHS=3 anchor at ``chosen_checkpoint_fraction`` from
    Phase 0 v4 §4.1 (the bystander-resolution picker).

Pass condition (ALL must hold — plan v5 §4.3a, tightened to 5% de-minimis):
  (a) For ≥ 1 of 20 (probe × q) pairs: ``|delta_g| > 1e-4`` AND ``kl > 0.01``.
  (b) ≤ 5% (≤ 1 of 20) pairs have ``|g_logp - b_logp| < 1e-6`` AND ``kl > 0.01``.

On fail, surfaces `epm:failure v1` reason `marker_logprob_path_still_broken`
and exits non-zero — the dispatcher MUST NOT spawn Phase 1 until this passes.

Usage (driven by dispatch_neg_geometry_504.py `--phase phase0p6_validate`):
    uv run python scripts/i504_phase_phase0p6.py \\
        --slab-root eval_results/issue_504 \\
        --phase0-pick-path eval_results/issue_504/phase0_calibration_v4.json \\
        --panel-json /tmp/i504-arm-to-n.json \\
        --bank-path data/issue_472/persona_bank.json \\
        --out-path eval_results/issue_504/phase0p6_validation_v4.json \\
        --hf-adapter-repo superkaiba1/explore-persona-space \\
        --hf-adapter-subfolder-prefix adapters/issue_504_v4/c504v4_smoke_eps3_seed42 \\
        --sentinel-path /workspace/logs/issue-504-phase0p6-results.json
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
from tempfile import TemporaryDirectory

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i504.phase_phase0p6")


# Plan v5 §4.3a Pass condition + §11 — de-minimis tightened in v5.
DELTA_G_NON_ZERO_NATS = 1e-4
BYTE_IDENTICAL_ABS_TOL = 1e-6
KL_DIAGNOSTIC_MIN_NATS = 0.01
BYTE_IDENTICAL_RATE_MAX = 0.05  # ≤ 1 of 20 = 5%
N_PROBES = 5
N_QUESTIONS = 4


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _resolve_adapter_dir(
    hf_repo: str,
    subfolder_prefix: str,
    chosen_frac: float,
    local_root: Path,
) -> Path:
    """Download the chosen-frac adapter from HF (or read locally) for the eval.

    Adapter HF path is built per plan v5 §4.0:
      ``<hf_repo>/<subfolder_prefix>/ckpt_frac{N}``
    where ``N`` is the chosen frac stringified per the canonical fraction
    set (0.08, 0.16, 0.33, 0.50, 0.75, 1.00). The local adapter dir under
    ``local_root`` is reused if the adapter weights already exist (resume-safe).

    Args:
        hf_repo: HF model repo (default `superkaiba1/explore-persona-space`).
        subfolder_prefix: subfolder prefix on HF (the v4 cell's training subdir).
        chosen_frac: the picked fraction (one of {0.08, 0.16, 0.33, 0.50, 0.75, 1.00}).
        local_root: filesystem root for the downloaded adapter dir.

    Returns:
        Absolute Path to a directory containing `adapter_model.safetensors` +
        `adapter_config.json`.

    Raises:
        FileNotFoundError: post-download, adapter weights are still missing
            on disk (Hub returned empty or wrong subfolder).
    """
    from huggingface_hub import snapshot_download

    # Canonical formatting matches plan v5 §4.0 path
    # `adapters/issue_504_v4/c504v4_smoke_eps3_seed42/ckpt_frac{N}`. {N} is
    # the canonical 2dp fraction (1.00 for chosen_frac=1.00).
    frac_token = "1.00" if abs(chosen_frac - 1.0) < 1e-6 else f"{chosen_frac:.2f}"
    subfolder = f"{subfolder_prefix.rstrip('/')}/ckpt_frac{frac_token}"
    local_dir = local_root / subfolder

    if not (local_dir / "adapter_model.safetensors").exists():
        local_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "[phase=phase0p6] downloading adapter from %s @ %s → %s",
            hf_repo,
            subfolder,
            local_dir,
        )
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=[f"{subfolder}/*"],
            local_dir=str(local_root),
            token=os.environ.get("HF_TOKEN"),
        )

    if not (local_dir / "adapter_model.safetensors").exists():
        raise FileNotFoundError(
            f"phase0p6: adapter weights missing at {local_dir} after Hub fetch. "
            f"Repo={hf_repo}, subfolder={subfolder}. Verify the Phase 0 §4.0 "
            f"pre-train uploaded the chosen-frac checkpoint via "
            f"`huggingface_hub.list_repo_files` (NOT the `hf` CLI — see "
            f".claude/rules/upload-policy.md)."
        )
    return local_dir


def _evaluate_pass_condition(checkpoint_payload: dict) -> dict:
    """Evaluate Phase 0.6 pass condition over the (probe × q) records.

    Reads ``checkpoint_payload["held_out"][persona][q]`` = {"g_logp", "b_logp",
    "delta_g", "kl", ...} and computes:

      - pass_a: ≥ 1 of 20 pairs has |delta_g| > 1e-4 AND kl > 0.01.
      - pass_b: ≤ 5% (≤ 1 of 20) pairs have |g_logp - b_logp| < 1e-6 AND kl > 0.01.

    Returns:
        {"pass_a": bool, "pass_b": bool, "byte_identical_rate": float,
         "n_delta_g_nonzero": int, "n_byte_identical": int, "n_total": int,
         "verdict": "PASS" | "FAIL"}.
    """
    n_total = 0
    n_dg_nonzero = 0
    n_byte_identical = 0
    for per_q in checkpoint_payload.get("held_out", {}).values():
        for leaf in per_q.values():
            n_total += 1
            dg = float(leaf["delta_g"])
            g_logp = float(leaf["g_logp"])
            b_logp = float(leaf["b_logp"])
            kl_val = leaf.get("kl")
            kl_float = float(kl_val) if kl_val is not None else 0.0
            if abs(dg) > DELTA_G_NON_ZERO_NATS and kl_float > KL_DIAGNOSTIC_MIN_NATS:
                n_dg_nonzero += 1
            if abs(g_logp - b_logp) < BYTE_IDENTICAL_ABS_TOL and kl_float > KL_DIAGNOSTIC_MIN_NATS:
                n_byte_identical += 1
    rate = n_byte_identical / n_total if n_total else 0.0
    pass_a = n_dg_nonzero >= 1
    pass_b = rate <= BYTE_IDENTICAL_RATE_MAX
    verdict = "PASS" if (pass_a and pass_b) else "FAIL"
    return {
        "pass_a": pass_a,
        "pass_b": pass_b,
        "byte_identical_rate": rate,
        "n_delta_g_nonzero": int(n_dg_nonzero),
        "n_byte_identical": int(n_byte_identical),
        "n_total": int(n_total),
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_504"))
    ap.add_argument(
        "--phase0-pick-path",
        type=Path,
        default=None,
        help=(
            "Path to phase0_calibration_v4.json (the v4 bystander-resolution "
            "pick). Default: <slab_root>/phase0_calibration_v4.json."
        ),
    )
    ap.add_argument(
        "--panel-json",
        type=Path,
        required=True,
        help=(
            "Phase 0.5 panel JSON ({'held_out_panel': [...], 'arm_to_positioned_n': "
            "{...}}). The first 5 personas in alphabetical order are sampled."
        ),
    )
    ap.add_argument(
        "--bank-path",
        type=Path,
        default=Path("data/issue_472/persona_bank.json"),
    )
    ap.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Output path. Default <slab_root>/phase0p6_validation_v4.json.",
    )
    ap.add_argument(
        "--hf-adapter-repo",
        default="superkaiba1/explore-persona-space",
        help="HF model repo carrying the v4 trajectory checkpoints (plan v5 §4.0).",
    )
    ap.add_argument(
        "--hf-adapter-subfolder-prefix",
        default="adapters/issue_504_v4/c504v4_smoke_eps3_seed42",
        help=(
            "HF subfolder prefix under --hf-adapter-repo containing the v4 "
            "EPOCHS=3 seed=42 trajectory checkpoints (each at "
            "`<prefix>/ckpt_frac<F>/`)."
        ),
    )
    ap.add_argument(
        "--adapter-local-root",
        type=Path,
        default=Path("/workspace/runs/issue_504/v4_anchor_download"),
        help="Local root for the downloaded adapter dir.",
    )
    ap.add_argument(
        "--source",
        default=None,
        help="Source persona. Defaults to villain (plan v5 §10).",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
    )
    ap.add_argument(
        "--max-lora-rank",
        type=int,
        default=8,
        help="LoRA rank pinned by Phase 0 (always 8 in v4/v5).",
    )
    ap.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.60,
        help="vLLM gpu_memory_utilization.",
    )
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=2560,
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sentinel-path", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=phase0p6_validate] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    pick_path = args.phase0_pick_path or args.slab_root / "phase0_calibration_v4.json"
    out_path = args.out_path or args.slab_root / "phase0p6_validation_v4.json"
    if not pick_path.exists():
        raise FileNotFoundError(
            f"phase0p6: Phase 0 v4 pick artifact missing at {pick_path}. "
            f"Run `--phase phase0_v4_reeval` first."
        )

    pick = json.loads(pick_path.read_text())
    if pick.get("verdict") != "pass":
        raise RuntimeError(
            f"phase0p6: Phase 0 v4 pick verdict={pick.get('verdict')!r}, "
            f"fallback_reason={pick.get('fallback_reason')!r}. Cannot validate "
            f"a non-existent anchor; the dispatcher should have routed to the "
            f"§4.2 EPOCHS=2 bisection before this script runs."
        )
    chosen_frac = float(pick["chosen_checkpoint_fraction"])
    chosen_epochs = int(pick["chosen_epochs"])
    chosen_source = pick.get("source") or args.source
    log.info(
        "[load] Phase 0 v4 pick: epochs=%d, chosen_frac=%g, source=%r",
        chosen_epochs,
        chosen_frac,
        chosen_source,
    )

    # ── Load panel + bank; pick the deterministic 5 probes × 4 questions. ──
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        get_train_eval_questions,
    )

    panel_payload = json.loads(args.panel_json.read_text())
    held_out_panel = panel_payload.get("held_out_panel", [])
    if not held_out_panel:
        raise RuntimeError(f"phase0p6: --panel-json {args.panel_json} has empty held_out_panel.")
    # Plan v5 §4.3a: alphabetical order; take first 5.
    probes_used = sorted(held_out_panel)[:N_PROBES]
    if len(probes_used) < N_PROBES:
        raise RuntimeError(
            f"phase0p6: held_out_panel has only {len(probes_used)} probes; need ≥ {N_PROBES}."
        )

    bank = load_persona_bank(args.bank_path)
    for p in probes_used:
        if p not in bank:
            raise KeyError(f"phase0p6: probe {p!r} missing from persona bank at {args.bank_path}.")
    eval_personas = {p: bank[p] for p in probes_used}

    _q_train, q_eval = get_train_eval_questions()
    questions_used_indices = list(range(N_QUESTIONS))
    if len(q_eval) < N_QUESTIONS:
        raise RuntimeError(
            f"phase0p6: q_eval has only {len(q_eval)} questions; need ≥ {N_QUESTIONS}."
        )
    questions_used = list(q_eval[:N_QUESTIONS])
    log.info(
        "[setup] probes=%s, n_questions=%d",
        probes_used,
        len(questions_used),
    )

    # ── Resolve adapter dir (download if needed). ──────────────────────────
    args.adapter_local_root.mkdir(parents=True, exist_ok=True)
    adapter_dir = _resolve_adapter_dir(
        hf_repo=args.hf_adapter_repo,
        subfolder_prefix=args.hf_adapter_subfolder_prefix,
        chosen_frac=chosen_frac,
        local_root=args.adapter_local_root,
    )
    log.info("[setup] adapter_dir=%s", adapter_dir)

    # ── Run the SAME fixed reader Phase 1 will use. ────────────────────────
    # `run_trajectory_eval` is the canonical reader (post fix #1 — per-batch
    # byte-identical guard wired into the same forward pass that computes
    # g_logp). We call it with a single checkpoint spec so the cost is bounded
    # at ~0.05 GPU-h (20 generations + 20 trained-logp + 20 base-logp + 20 KL).
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        BASE_MODEL,
        SOURCE_PERSONA,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        run_trajectory_eval,
    )

    effective_source = chosen_source or SOURCE_PERSONA
    if effective_source not in bank:
        raise KeyError(f"phase0p6: source {effective_source!r} missing from persona bank.")
    checkpoint_specs = [
        {
            "frac": chosen_frac,
            "step": pick.get("chosen_checkpoint_steps"),
            "adapter_path": str(adapter_dir),
        }
    ]

    # Run through `run_trajectory_eval` over a tmpdir so the rig's normal
    # trajectory.json + partial.json paths don't clobber Phase 0/1 outputs. We
    # parse the trajectory.json structure from the returned tmp file.
    with TemporaryDirectory(prefix="phase0p6_") as td:
        out_traj = Path(td) / "phase0p6_trajectory.json"
        log.info(
            "[phase=phase0p6_eval] running fixed reader on %d probes × %d questions × 1 ckpt → %s",
            len(probes_used),
            len(questions_used),
            out_traj,
        )
        # The rig's per-batch byte-identical guard (fix #1) is automatically
        # wired in `run_trajectory_eval`; if the marker-logprob path is still
        # broken, this raises MarkerLogprobPathReadingFromBaseError and exits
        # non-zero — the dispatcher catches it and surfaces epm:failure v1
        # `failure_class: code, reason: marker_logprob_path_still_broken`.
        run_trajectory_eval(
            cell_slug="phase0p6_validate",
            seed=args.seed,
            checkpoint_specs=checkpoint_specs,
            eval_personas=eval_personas,
            eval_questions=questions_used,
            source=effective_source,
            source_prompt=bank[effective_source],
            out_path=out_traj,
            base_model=BASE_MODEL,
            max_new_tokens=args.max_new_tokens,
            max_lora_rank=max(8, args.max_lora_rank),
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            compute_kl=True,
        )
        trajectory = json.loads(out_traj.read_text())

    # ── Phase 0.6 pass condition. ──────────────────────────────────────────
    if not trajectory.get("checkpoints"):
        raise RuntimeError(
            "phase0p6: trajectory has no checkpoint payload (run_trajectory_eval "
            "silently dropped the single ckpt). Investigate the rig before "
            "trusting any verdict."
        )
    ck = trajectory["checkpoints"][0]
    cond = _evaluate_pass_condition(ck)

    # Build the per-pair audit table for the artifact (plan v5 §4.3a output schema).
    results_per_pair: list[dict] = []
    for probe in probes_used:
        per_q = ck["held_out"][probe]
        for q_idx, q in enumerate(questions_used):
            leaf = per_q[q]
            results_per_pair.append(
                {
                    "probe": probe,
                    "question_idx": q_idx,
                    "g_logp": float(leaf["g_logp"]),
                    "b_logp": float(leaf["b_logp"]),
                    "delta_g": float(leaf["delta_g"]),
                    "kl": float(leaf["kl"]) if leaf.get("kl") is not None else None,
                    "byte_identical": (
                        abs(float(leaf["g_logp"]) - float(leaf["b_logp"])) < BYTE_IDENTICAL_ABS_TOL
                        and (leaf.get("kl") is not None)
                        and float(leaf["kl"]) > KL_DIAGNOSTIC_MIN_NATS
                    ),
                }
            )

    payload = {
        "version": 4,
        "anchor_checkpoint": (
            f"hf://{args.hf_adapter_repo}/"
            f"{args.hf_adapter_subfolder_prefix}/ckpt_frac{chosen_frac:.2f}"
        ),
        "anchor_local_path": str(adapter_dir),
        "chosen_epochs": chosen_epochs,
        "chosen_checkpoint_fraction": chosen_frac,
        "probes_used": probes_used,
        "questions_used_indices": questions_used_indices,
        "results_per_pair": results_per_pair,
        "pass_a": cond["pass_a"],
        "pass_b": cond["pass_b"],
        "byte_identical_rate": cond["byte_identical_rate"],
        "n_delta_g_nonzero": cond["n_delta_g_nonzero"],
        "n_byte_identical": cond["n_byte_identical"],
        "n_total": cond["n_total"],
        "verdict": cond["verdict"],
        "thresholds": {
            "delta_g_non_zero_nats": DELTA_G_NON_ZERO_NATS,
            "byte_identical_abs_tol": BYTE_IDENTICAL_ABS_TOL,
            "kl_diagnostic_min_nats": KL_DIAGNOSTIC_MIN_NATS,
            "byte_identical_rate_max": BYTE_IDENTICAL_RATE_MAX,
        },
        "source": effective_source,
        "seed": args.seed,
        "hostname": socket.gethostname(),
        "git_commit": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    log.info(
        "[phase=phase0p6_done] verdict=%s, pass_a=%s, pass_b=%s, "
        "byte_identical_rate=%.4f (%d/%d) — wrote %s",
        cond["verdict"],
        cond["pass_a"],
        cond["pass_b"],
        cond["byte_identical_rate"],
        cond["n_byte_identical"],
        cond["n_total"],
        out_path,
    )

    if args.sentinel_path is not None:
        args.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        args.sentinel_path.write_text(
            json.dumps(
                {
                    "sentinel_schema_version": 1,
                    "kind": "epm:progress",
                    "version": 1,
                    "task_id": 504,
                    "phase": "phase0p6_validate",
                    "by": "i504_phase_phase0p6",
                    "ts": datetime.now(UTC).isoformat(),
                    "note": json.dumps(
                        {
                            "verdict": cond["verdict"],
                            "pass_a": cond["pass_a"],
                            "pass_b": cond["pass_b"],
                            "byte_identical_rate": cond["byte_identical_rate"],
                            "n_byte_identical": cond["n_byte_identical"],
                            "n_total": cond["n_total"],
                            "out_path": str(out_path),
                        }
                    ),
                },
                indent=2,
            )
        )

    # FAIL → non-zero exit so the dispatcher's subprocess.run(check=True)
    # raises and the orchestrator posts epm:failure v1.
    if cond["verdict"] != "PASS":
        log.error(
            "[phase=phase0p6_validate] FAIL — verdict=%s, pass_a=%s, pass_b=%s, "
            "byte_identical_rate=%.4f. Marker-logprob path STILL BROKEN; do NOT "
            "spawn Phase 1. See plan v5 §4.3a + §11 for the failure modes.",
            cond["verdict"],
            cond["pass_a"],
            cond["pass_b"],
            cond["byte_identical_rate"],
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
