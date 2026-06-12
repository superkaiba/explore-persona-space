# ruff: noqa: RUF002, RUF003  # research code uses Greek letters (Δ), × and − legitimately
"""Task #597 dispatcher — training dynamics of source implantation vs bystander leakage.

Pipeline (smoke = sweep with one cell via ``--only-source villain --smoke``;
same dispatcher, same subprocess shape, same env injection, same logging
surface, same teardown — the smoke knobs only SCALE each phase):

  Phase P (preflight, CPU): in-process marker assert; question disjointness;
      bystander-assignment → panel-name mapping (trained-negative labels);
      700-row pool fetch (pinned revision) → order-preserving 200-positive
      filter → BLOCKING token-id probe-row identity assert; adapter-config
      parity vs a downloaded Arm A capend checkpoint; pos-pool upload.

  Phase 0 (vLLM subprocess): fixed probe-row generation — 25 contexts × 50
      eval_50 questions, base greedy R (cap 1024). One JSON, immutable after.

  Phase S (HF subprocess — HARD GATE, #534): off-line eval path must
      reproduce #480's in-loop band-stop read (villain capend ckpt-20/-40)
      within 1 nat trained / 0.1 nat base. FAIL → no sweep.

  Per-cell loop (one cell == one source, sequential per shard):
      trainB (in-process train_lora, max_steps=528, save_steps=4 + grid
      prune; SKIPPED when the complete in-budget ladder + trajectory already
      exist — recovery relaunches are idempotent; a ladder run-id is stamped
      at train end / adoption so panel_probe's resume-skip is provenance-
      gated) → Gate S re-application on the FIRST Arm B source TRAINED IN
      THIS PROCESS vs its own fresh trajectory (ALWAYS runs per shard —
      decoupled from --skip-arm-a-gate; #518 reachability class) → Arm B
      ladder upload (fail-loud) → panel probe Arm B (HF subprocess) →
      emission anchors Arm B (vLLM subprocess) → Arm A ladder download
      (per-file; never snapshot_download allow_patterns) → panel probe
      Arm A → emission anchors Arm A → raw uploads (folder-level commits) →
      local cleanup → per-source sentinel (poll_pipeline schema, kind
      epm:progress — bare cell dicts inside the issue-597-*.json glob are
      skipped as malformed by the poller forever).

  Final sentinel: /workspace/logs/issue-597-epm_results-<epoch>-<pid>.json
  (poll_pipeline schema: sentinel_schema_version=1, kind=epm:results, ...;
  pid suffix prevents same-second shard collisions), then the terminal
  ``[phase=done]`` line.

GPU sharding (plan §9 — 6 sources over 4 GPUs, ONE pod): launch one process
per shard with the GPU pinned BOTH in the LAUNCHER env (an import-time cuInit
would defeat a late in-process pin — gotchas.md) AND via --gpu (threaded into
TrainLoraConfig.gpu_id so train_lora's unconditional CUDA_VISIBLE_DEVICES
clobber re-asserts the SAME physical index — the #557 class; without it every
shard's training + downstream subprocesses co-locate on physical GPU 0):

    # one-time shared phases (Phase 0 + Gate S) on GPU 0:
    CUDA_VISIBLE_DEVICES=0 nohup uv run python \\
        scripts/issue_597/dispatch_leakage_dynamics_597.py \\
        --recipe pos_only_dynamics --stop-after-gate > logs/i597_gate.log 2>&1 < /dev/null &
    # then per-shard sweeps (skip the SHARED phases only — the per-shard
    # first-Arm-B-source Gate S re-application always runs):
    CUDA_VISIBLE_DEVICES=0 nohup uv run python ... --gpu 0 \\
        --sources villain,comedian --skip-probe-rows --skip-arm-a-gate ... &
    CUDA_VISIBLE_DEVICES=1 nohup uv run python ... --gpu 1 \\
        --sources assistant,qwen_default --skip-probe-rows --skip-arm-a-gate ... &

--recipe contrastive_dense_early (#597 follow-up `dense-early-contrastive-grid`,
plan v3): fresh retrain of the 6 contrastive cells on the SAME pinned 700-row
pools with a dense early checkpoint grid (C_GRID = {2..40:2} ∪ {44..60:4},
save_steps=2, save-driven halt after the step-60 save — max_steps stays 528 so
the cosine + warmup schedule is identical to #480 for steps 1–60). Phase 0 is
replaced by a revision-pinned fetch of the parent's probe_rows.json; the Arm A
ladder leg and emission anchors are skipped (plan v3 §2.3); a CPU parity gate
joins the dense panel reads at steps 20/40/60 against the parent's committed
armA panel trajectories (BLOCKING at step 20: ±2 nat on source Δ AND
TN-median in ≥5/6 sources). Smoke = sweep with one cell:

    uv run python scripts/issue_597/dispatch_leakage_dynamics_597.py \\
        --recipe contrastive_dense_early --only-source villain --smoke
    # production (plan §10): --recipe contrastive_dense_early --seed 42 --gpu 0

Pod-side discipline (CLAUDE.md):
- NEVER shells out to scripts/task.py (branch-guard would refuse).
- Every subprocess.* call passes env={**os.environ}; load_dotenv() at module top.
- [phase=...] log lines, terminating in [phase=done] on graceful exit
  (poll_pipeline contract); per-cell completion lines never carry that token.
- EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1 (this dispatcher owns its uploads).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("issue_597.dispatch")

PKG = "explore_persona_space.experiments.leakage_dynamics_597"

# ── reuse constants (provenance: scripts/issue_480/dispatch_marker_480.py) ──
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_WRONG_CLAIMS_SUBDIR = "issue411_sycophancy_cosine_gradient/data/wrong_claims"
HF_TRAIN_POOL_SUBDIR = "issue480_marker_payload_swap/train_pools"
HF_BYSTANDER_ASSIGNMENT = "issue480_marker_payload_swap/inputs/bystander_assignment.json"
TRAIN_POOL_REVISION = "3c8fecb937c81c13036a9697be1e4e716755321e"
TRAIN_POOL_EXPECTED_ROWS = 700
BAND_STOP_LR = 5e-6
BAND_STOP_EPOCH_CAP = 12
BAND_STOP_PROBE_EVERY_STEPS = 5
SENTINEL_SCHEMA_VERSION = 1

# Arm A reference trajectory JSONs (in git — present on the pod checkout).
ARM_A_TRAJ_DIR = Path("eval_results/issue_480/band-stopped-anchor-rerun/trajectories")

# Arm A capend adapter geometry reference for the parity preflight.
ARM_A_PARITY_CONFIG = (
    "adapters/issue_480_band_stop/villain_seed42_capend/checkpoint-20/adapter_config.json"
)

# train_lora's _DEFAULT_LORA_TARGETS (sft.py function-local literal), mirrored
# for the parity check — same mirror dispatch_marker_480.py carries.
_TRAIN_LORA_DEFAULT_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Retry-with-backoff for HF Hub downloads (dispatcher silent-death hardening).
_HF_RETRY_SLEEPS = (30, 60, 120)

# ── #597 follow-up `dense-early-contrastive-grid` (plan v3) ──────────────────
# Phase 0 is SKIPPED for the dense recipe: probe rows are the parent run's own
# Phase-U upload, fetched at the PINNED revision (content-identity check (f)).
HF_597_PROBE_ROWS_FILE = "issue597_leakage_dynamics/inputs/probe_rows.json"
PROBE_ROWS_REVISION = "8d2f79030e365180c7d32755cda34d34a25aed18"
# Follow-up artifacts live under their own label dir (eval_results/issue_597/
# <followup_label>/ — same-issue follow-up routing convention).
DENSE_SLAB_SUBDIR = "dense-early-contrastive-grid"
# Plan-v1 §13 sanctioned read-only deviation: 2-step in-loop source
# corroboration alongside the 2-step checkpoint grid (Arm B used 5).
DENSE_BAND_PROBE_EVERY_STEPS = 2
# Parent contrastive (Arm A) panel trajectories — the parity-gate reference
# (in git on the pod checkout; values quoted in plan v3 §7).
ARM_A_PANEL_DIR = Path("eval_results/issue_597/panel_trajectories/armA")
# Parity gate (plan v3 §7): BLOCKING at step 20 only; 40/60 diagnostic.
PARITY_STEPS = (20, 40, 60)
PARITY_BLOCKING_STEP = 20
PARITY_TOL_NATS = 2.0  # ≈2 effective steps at the measured 0.45–1.0 nat/step ramp
PARITY_CATASTROPHIC_NATS = 5.0
PARITY_LOCKSTEP_INVERSION_RATIO = 0.5  # TN tracking source — the pos-only signature
PARITY_MIN_PASS_SOURCES = 5
PARITY_REGISTERED_N_SOURCES = 6
PARITY_BASE_DIAG_TOL_NATS = 0.1  # plan §12.6 logged diagnostic, never a gate


@dataclass(frozen=True)
class RunParams:
    """Smoke-vs-sweep scale knobs, threaded through EVERY phase (PASS_UNIFIED).

    The smoke is the sweep with one cell + scaled-down parameters; every
    phase's checkpoint / anchor / question subset derives from THIS object —
    no phase re-enumerates a full registered grid on its own (#546 round-1
    failure class).
    """

    smoke: bool
    b_max_steps: int
    b_save_steps: int
    b_grid: tuple[int, ...]
    a_steps: tuple[int, ...]
    anchor_steps: tuple[int, ...]
    limit_questions: int | None
    hf_suffix: str  # "" for the sweep, "_smoke" for smoke uploads


def make_run_params(smoke: bool) -> RunParams:
    from explore_persona_space.experiments.leakage_dynamics_597 import (
        A_GRID,
        ANCHOR_STEPS,
        ARM_B_MAX_STEPS,
        ARM_B_SAVE_STEPS,
        B_GRID,
    )

    if smoke:
        # Smoke scale: 24 optimizer steps (>= the band callback's min_steps=20
        # so the in-loop trajectory gets real probes at steps 5..20 and the
        # Gate S re-application on Arm B has its step-20 reference), the first
        # two Arm A checkpoints, one anchor step, 5 questions.
        return RunParams(
            smoke=True,
            b_max_steps=24,
            b_save_steps=ARM_B_SAVE_STEPS,
            b_grid=(4, 8, 12, 16, 20, 24),
            a_steps=(20, 40),
            anchor_steps=(20,),
            limit_questions=5,
            hf_suffix="_smoke",
        )
    return RunParams(
        smoke=False,
        b_max_steps=ARM_B_MAX_STEPS,
        b_save_steps=ARM_B_SAVE_STEPS,
        b_grid=B_GRID,
        a_steps=A_GRID,
        anchor_steps=ANCHOR_STEPS,
        limit_questions=None,
        hf_suffix="",
    )


@dataclass(frozen=True)
class DenseRunParams:
    """Smoke-vs-sweep scale knobs for ``--recipe contrastive_dense_early``.

    Same PASS_UNIFIED contract as :class:`RunParams`: the smoke is the sweep
    with one cell + scaled-down knobs, and every phase's checkpoint / gate /
    question subset derives from THIS object — no phase re-enumerates a full
    registered grid on its own. ``max_steps`` is deliberately NOT a knob:
    it stays 528 in BOTH regimes (the save-driven halt is what scales —
    schedule identity for steps 1–halt is the design invariant, plan v3 §2).
    """

    smoke: bool
    halt_step: int
    save_steps: int
    c_grid: tuple[int, ...]
    probe_steps: tuple[int, ...]
    gate_step: int
    limit_questions: int | None
    hf_suffix: str  # "" for the sweep, "_smoke" for smoke uploads


def make_dense_run_params(smoke: bool) -> DenseRunParams:
    from explore_persona_space.experiments.leakage_dynamics_597 import (
        ARM_C_HALT_STEP,
        ARM_C_SAVE_STEPS,
        C_GRID,
    )

    if smoke:
        # Smoke scale (plan v3 §3): halt at step 12, grid {2..12:2}, 5
        # questions, 2 probed checkpoints (first + last — exercises the
        # multi-checkpoint loop AND the end-of-ladder hot-swap invariant).
        # Gate at step 12: the in-loop band probe records every 2 steps so
        # step 12 carries a reference; step 20 doesn't exist under the halt.
        return DenseRunParams(
            smoke=True,
            halt_step=12,
            save_steps=ARM_C_SAVE_STEPS,
            c_grid=(2, 4, 6, 8, 10, 12),
            probe_steps=(2, 12),
            gate_step=12,
            limit_questions=5,
            hf_suffix="_smoke",
        )
    return DenseRunParams(
        smoke=False,
        halt_step=ARM_C_HALT_STEP,
        save_steps=ARM_C_SAVE_STEPS,
        c_grid=C_GRID,
        probe_steps=C_GRID,
        gate_step=20,
        limit_questions=None,
        hf_suffix="",
    )


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


def _run_subprocess(cmd: list[str], phase: str) -> None:
    """Run a phase subprocess with explicit env passthrough; fail loud."""
    log.info("[phase=%s] spawning: %s", phase, " ".join(cmd))
    subprocess.run(cmd, env={**os.environ}, check=True)


def _hf_download_with_retry(repo_id: str, filename: str, **kwargs) -> str:
    """hf_hub_download with 3-attempt backoff (transient-blip hardening)."""
    from huggingface_hub import hf_hub_download

    last_err: Exception | None = None
    for attempt, sleep_s in enumerate((0, *_HF_RETRY_SLEEPS)):
        if sleep_s:
            log.warning(
                "[phase=preflight] retrying %s in %ds (attempt %d): %s",
                filename,
                sleep_s,
                attempt + 1,
                last_err,
            )
            time.sleep(sleep_s)
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"hf_hub_download failed after retries: {repo_id}/{filename}") from last_err


def ensure_train_pool(local_path: Path, source: str) -> Path:
    """Fetch the parent's 700-row pool for ``source`` at the PINNED revision.

    Pattern reused from ``dispatch_marker_480._ensure_train_pool`` (fail loud
    on a row count != 700 — a short pool means the wrong artifact resolved).
    """
    if not local_path.exists():
        cached = _hf_download_with_retry(
            repo_id=HF_DATA_REPO,
            filename=f"{HF_TRAIN_POOL_SUBDIR}/{source}_train_pool.jsonl",
            repo_type="dataset",
            revision=TRAIN_POOL_REVISION,
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, local_path)
    n_rows = sum(1 for line in local_path.read_text().splitlines() if line.strip())
    if n_rows != TRAIN_POOL_EXPECTED_ROWS:
        raise RuntimeError(
            f"train pool {local_path} has {n_rows} rows, expected "
            f"{TRAIN_POOL_EXPECTED_ROWS} (revision {TRAIN_POOL_REVISION})"
        )
    log.info("[phase=preflight] train pool ready at %s (%d rows)", local_path, n_rows)
    return local_path


def ensure_wrong_claim_pool(local_path: Path, kind: str) -> Path:
    """Fetch a #411 wrong-claim Q pool (train_200 | eval_50) if missing locally."""
    if local_path.exists():
        return local_path
    cached = _hf_download_with_retry(
        repo_id=HF_DATA_REPO,
        filename=f"{HF_WRONG_CLAIMS_SUBDIR}/{kind}.jsonl",
        repo_type="dataset",
    )
    local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, local_path)
    log.info("[phase=preflight] wrong-claim pool ready at %s", local_path)
    return local_path


def ensure_pinned_probe_rows(local_path: Path) -> Path:
    """Fetch the PARENT run's probe_rows.json at the PINNED revision (plan v3 §3).

    The dense recipe SKIPS Phase 0: the probe rows are the parent run's own
    Phase-U upload, revision-pinned (the content-identity mechanism — reuse
    check (f)). Shape-asserted to the PRODUCTION scale (25 contexts × 50
    questions) on every call, fetched or reused: the smoke caps questions at
    probe time via ``--limit-questions``, never by fetching a different
    artifact.
    """
    if not local_path.exists():
        cached = _hf_download_with_retry(
            repo_id=HF_DATA_REPO,
            filename=HF_597_PROBE_ROWS_FILE,
            repo_type="dataset",
            revision=PROBE_ROWS_REVISION,
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, local_path)
    hdr = json.loads(local_path.read_text())
    if (
        hdr.get("schema") != "i597_probe_rows_v1"
        or hdr.get("n_contexts") != 25
        or hdr.get("n_questions") != 50
    ):
        raise RuntimeError(
            f"pinned probe rows at {local_path} have unexpected shape "
            f"(schema={hdr.get('schema')!r}, contexts={hdr.get('n_contexts')}, "
            f"questions={hdr.get('n_questions')}), expected (i597_probe_rows_v1, 25, 50) — "
            f"wrong artifact resolved at revision {PROBE_ROWS_REVISION}; refusing to probe."
        )
    log.info(
        "[phase=p0_probe_rows] pinned probe rows ready at %s (rev %s)",
        local_path,
        PROBE_ROWS_REVISION[:12],
    )
    return local_path


def _load_wrong_claims(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line)["wrong_claim"])
    return out


def assert_question_disjointness(train_pool: Path, eval_pool: Path) -> None:
    """Plan Phase P: train_200 wrong-claims ∩ eval_50 wrong-claims = ∅."""
    train_qs = _load_wrong_claims(train_pool)
    eval_qs = _load_wrong_claims(eval_pool)
    if len(train_qs) != 200 or len(eval_qs) != 50:
        raise RuntimeError(
            f"question pool sizes drifted: train={len(train_qs)} (want 200), "
            f"eval={len(eval_qs)} (want 50)"
        )
    overlap = set(train_qs) & set(eval_qs)
    if overlap:
        raise RuntimeError(
            f"train/eval wrong-claim pools overlap on {len(overlap)} question(s): "
            f"{sorted(overlap)[:3]}..."
        )
    log.info("[phase=preflight] question disjointness OK (200 train / 50 eval, overlap 0)")


def load_trained_negative_map(cache_path: Path) -> dict[str, list[str]]:
    """Download bystander_assignment.json and map system prompts → panel names.

    Plan Phase P: exact prompt-string match against ``EVAL_PERSONAS_24``;
    fail loud on any unmapped prompt; per cell assert the negatives exclude
    that cell's own source (the mapping labels the trained-negative vs
    held-out bystander split in analysis).
    """
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    if not cache_path.exists():
        cached = _hf_download_with_retry(
            repo_id=HF_DATA_REPO, filename=HF_BYSTANDER_ASSIGNMENT, repo_type="dataset"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, cache_path)
    assignment = json.loads(cache_path.read_text())

    prompt_to_name: dict[str, str] = {}
    for name, prompt in EVAL_PERSONAS_24.items():
        if prompt in prompt_to_name:
            raise RuntimeError(f"EVAL_PERSONAS_24 prompts not unique: {name!r}")
        prompt_to_name[prompt] = name

    neg_map: dict[str, list[str]] = {}
    for source, info in assignment.items():
        names: list[str] = []
        for sp in info["system_prompts"]:
            if sp not in prompt_to_name:
                raise RuntimeError(
                    f"bystander prompt for source {source!r} not in EVAL_PERSONAS_24: {sp[:80]!r}"
                )
            names.append(prompt_to_name[sp])
        if source in names:
            raise RuntimeError(
                f"trained-negative set for source {source!r} contains the source itself: {names}"
            )
        if len(set(names)) != 2:
            raise RuntimeError(f"expected 2 distinct negatives for {source!r}, got {names}")
        neg_map[source] = names
        log.info("[phase=preflight] trained negatives for %s: %s", source, names)

    # Cross-assert vs the package constant the OFF-POD analysis consumes
    # (analyze.py reads TRAINED_NEGATIVES; a drifted assignment artifact must
    # fail HERE, at preflight, not silently skew the H2 split at analysis).
    from explore_persona_space.experiments.leakage_dynamics_597 import TRAINED_NEGATIVES

    derived = {s: frozenset(v) for s, v in neg_map.items()}
    pinned = {s: frozenset(v) for s, v in TRAINED_NEGATIVES.items()}
    if derived != pinned:
        raise RuntimeError(
            "bystander_assignment.json drifted from the pinned TRAINED_NEGATIVES "
            f"constant: derived={derived} pinned={pinned} — fix the constant or the "
            "artifact before launching (the off-pod analysis keys on the constant)."
        )
    return neg_map


def effective_shard_gpu(gpu_arg: int | None) -> int:
    """Resolve the shard's ``--gpu`` into the ``TrainLoraConfig.gpu_id`` pin.

    ``train_lora`` UNCONDITIONALLY clobbers ``CUDA_VISIBLE_DEVICES`` with
    ``cfg.gpu_id`` (sft.py), and this dispatcher performs NO CUDA init before
    ``train_lora`` — so the clobber WINS over the launcher's exported CVD
    (#557 class; round-1 review blocker). Threading the shard's ``--gpu``
    here makes the clobber re-assert the SAME physical index the launcher
    exported, and every downstream SUBPROCESS (panel_probe's second 7B, the
    emission vLLM at gpu_memory_utilization=0.85) inherits the correct CVD.
    No ``--gpu`` (single-process / smoke launch) → physical GPU 0.
    """
    return gpu_arg if gpu_arg is not None else 0


def _pos_only_train_cfg(
    source: str,
    seed: int,
    max_length: int,
    traj_path: Path,
    *,
    max_steps: int,
    save_steps: int,
    gpu_id: int = 0,
):
    """Arm B TrainLoraConfig — the #480 ``_band_stop_train_cfg`` clone.

    Identical in every field to the realized Arm A launch config (enforced by
    :func:`assert_pos_only_adapter_parity` against the downloaded capend
    adapter_config.json) EXCEPT the documented #597 deltas: ``max_steps``
    (exactly matched 528-step cosine schedule; epochs alone would give 533+),
    ``save_steps`` (4, for the fine early grid — pruned to B_GRID by
    ``CheckpointGridPruneCallback``), ``run_name``, the trajectory path, and
    ``gpu_id`` (the shard's physical GPU — see :func:`effective_shard_gpu`;
    NOT a training-recipe field, it never reaches the adapter geometry).
    The training DATA difference (200-positive pool) is the manipulated
    variable and is passed separately at the train_lora call site.
    """
    from explore_persona_space.experiments.leakage_dynamics_597 import IM_END_ID, MARKER_TEXT
    from explore_persona_space.train.sft import TrainLoraConfig

    return TrainLoraConfig(
        gpu_id=gpu_id,
        epochs=BAND_STOP_EPOCH_CAP,  # superseded by max_steps; kept for field parity
        lr=BAND_STOP_LR,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        batch_size=4,
        grad_accum=4,  # effective batch 16
        max_length=max_length,
        warmup_ratio=0.05,
        seed=seed,
        run_name=f"issue597_posonly_{source}_seed{seed}",
        report_to="wandb",
        save_strategy="steps",
        save_steps=save_steps,
        save_only_model=True,
        gradient_checkpointing=True,
        packing=False,
        marker_only_loss=True,
        marker_text=MARKER_TEXT,
        marker_tail_tokens=0,
        # Kept for cfg parity with Arm A; the branch never fires on the
        # positive-only pool (no marker-less rows exist).
        marker_suppress_at_post_response_slot=True,
        marker_im_end_token_id=IM_END_ID,
        marker_band_stop=True,
        marker_band_log_only=True,  # full ramp — never stops, logs the 5-step trajectory
        marker_band_eval_every_steps=BAND_STOP_PROBE_EVERY_STEPS,
        marker_band_trajectory_path=str(traj_path),
        # This dispatcher uploads the ladder itself, FAIL-LOUD, before any
        # deletion; train_lora's best-effort soft-fail upload is disabled.
        hf_upload=False,
        max_steps=max_steps,
    )


def _dense_train_cfg(
    source: str,
    seed: int,
    max_length: int,
    traj_path: Path,
    *,
    save_steps: int | None = None,
    gpu_id: int = 0,
):
    """Arm C (dense-early contrastive) TrainLoraConfig — plan v3 §3.

    Cloned from :func:`_pos_only_train_cfg` (itself the realized #480
    ``_band_stop_train_cfg``), so every RECIPE field — lr 5e-6, r=32/α=64
    rsLoRA 7-module, eff. batch 16, warmup 0.05, marker-only loss,
    ``max_steps=528`` — is inherited verbatim. The only deltas are
    instrumental (plan v3 §2): ``save_steps=2`` (dense grid, pruned to
    C_GRID by ``CheckpointGridPruneCallback``), ``marker_band_eval_every_
    steps=2`` (plan-v1 §13 sanctioned read-only deviation — 2-step in-loop
    source corroboration), ``run_name``, and the trajectory path. The halt
    after step 60 is a CALLBACK (``HaltAfterStepCallback``), NOT a
    ``max_steps`` change — schedule identity for steps 1–60 is the design
    invariant (unit-pinned by the lr(step) identity test). The training DATA
    (the full 700-row contrastive pool) is passed at the train_lora call
    site, unchanged from #480.
    """
    from explore_persona_space.experiments.leakage_dynamics_597 import ARM_C_SAVE_STEPS

    cfg = _pos_only_train_cfg(
        source,
        seed,
        max_length,
        traj_path,
        max_steps=528,  # the #480 schedule total — NEVER scaled (halt is save-driven)
        save_steps=save_steps if save_steps is not None else ARM_C_SAVE_STEPS,
        gpu_id=gpu_id,
    )
    return replace(
        cfg,
        run_name=f"issue597_densegrid_{source}_seed{seed}",
        marker_band_eval_every_steps=DENSE_BAND_PROBE_EVERY_STEPS,
    )


def assert_pos_only_adapter_parity(max_length: int, cfg=None) -> dict:
    """Adapter-config parity preflight vs a downloaded Arm A capend checkpoint.

    Single-variable-change enforcement (plan Phase P): the fresh-training
    ``TrainLoraConfig`` (Arm B, or Arm C when ``cfg`` is the dense builder's
    probe config) must produce identical PEFT geometry (r, α, dropout,
    rsLoRA, target_modules, modules_to_save=∅) to Arm A's published
    checkpoints. Fails loud pre-GPU. Pattern reused from
    ``dispatch_marker_480._assert_band_stop_adapter_parity``.
    """
    cached = _hf_download_with_retry(repo_id=HF_MODEL_REPO, filename=ARM_A_PARITY_CONFIG)
    with open(cached) as f:
        parent = json.load(f)

    if cfg is None:
        cfg = _pos_only_train_cfg(
            source="_parity_probe",
            seed=42,
            max_length=max_length,
            traj_path=Path("/tmp/_i597_parity_probe_trajectory.json"),
            max_steps=528,
            save_steps=4,
        )
    expected_targets = sorted(cfg.lora_targets or _TRAIN_LORA_DEFAULT_TARGETS)
    checks: dict[str, tuple[object, object]] = {
        "r": (parent.get("r"), cfg.lora_r),
        "lora_alpha": (parent.get("lora_alpha"), cfg.lora_alpha),
        "lora_dropout": (parent.get("lora_dropout"), cfg.lora_dropout),
        "use_rslora": (parent.get("use_rslora"), True),
        "target_modules": (sorted(parent.get("target_modules") or []), expected_targets),
        "modules_to_save": (parent.get("modules_to_save"), None),
    }
    mismatches = {k: v for k, v in checks.items() if v[0] != v[1]}
    for key, (parent_val, ours) in checks.items():
        log.info(
            "[phase=preflight] adapter-config parity %s: armA=%s armB=%s %s",
            key,
            parent_val,
            ours,
            "MISMATCH" if key in mismatches else "OK",
        )
    if mismatches:
        raise RuntimeError(
            f"Arm B adapter-config parity FAILED vs {HF_MODEL_REPO}/{ARM_A_PARITY_CONFIG}: "
            f"{mismatches} — the regime swap is no longer single-variable; refusing to launch."
        )
    log.info("[phase=preflight] adapter-config parity vs Arm A PASSED (%d keys)", len(checks))
    return {k: v[1] for k, v in checks.items()}


def download_arm_a_ladder(source: str, steps: tuple[int, ...], dest_root: Path) -> Path:
    """Per-file download of the Arm A capend checkpoints for ``steps``.

    NEVER ``snapshot_download(allow_patterns=...)`` — on this >8k-file repo it
    silently returns 0 files for prefixes in the truncated siblings tail
    (feedback_snapshot_download_siblings_truncation). Enumerate via
    ``list_repo_files`` (the plan-§12-verified path) + ``hf_hub_download``
    per file, with per-file completion log lines.
    """
    from huggingface_hub import list_repo_files

    from explore_persona_space.experiments.leakage_dynamics_597 import ARM_A_HF_ADAPTER_ROOT

    prefix = f"{ARM_A_HF_ADAPTER_ROOT}/{source}_seed42_capend"
    dest = dest_root / source
    wanted_dirs = {f"{prefix}/checkpoint-{s}" for s in steps}
    have_all = all((dest / f"checkpoint-{s}" / "adapter_config.json").exists() for s in steps)
    if have_all:
        log.info(
            "[phase=download_a_%s] all %d checkpoints already local; skipping", source, len(steps)
        )
        return dest

    all_files = list_repo_files(HF_MODEL_REPO)
    to_fetch = [f for f in all_files if any(f.startswith(d + "/") for d in wanted_dirs)]
    if not to_fetch:
        raise RuntimeError(
            f"no files found under {prefix}/checkpoint-{{{','.join(map(str, steps))}}} on "
            f"{HF_MODEL_REPO} — Arm A reuse premise broken."
        )
    # Every requested step must resolve at least an adapter_config + weights.
    for s in steps:
        d = f"{prefix}/checkpoint-{s}/"
        if not any(f.startswith(d) for f in to_fetch):
            raise RuntimeError(f"Arm A checkpoint-{s} missing on the Hub under {prefix}")
    log.info(
        "[phase=download_a_%s] fetching %d files for %d checkpoints",
        source,
        len(to_fetch),
        len(steps),
    )
    for i, fname in enumerate(to_fetch):
        cached = _hf_download_with_retry(repo_id=HF_MODEL_REPO, filename=fname)
        rel = Path(fname).relative_to(prefix)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, target)
        log.info("[phase=download_a_%s] (%d/%d) %s", source, i + 1, len(to_fetch), rel)
    return dest


def upload_dir_fail_loud(local_dir: Path, repo_id: str, repo_type: str, path_in_repo: str) -> str:
    """ONE-commit folder upload via the shared hub helper; raise on failure.

    Folder-level commits keep the sweep far under the HF 256-commits/hr cap
    (upload-policy rule); ``_upload`` verifies via ``list_repo_files`` before
    returning.
    """
    from explore_persona_space.orchestrate.hub import _upload

    hub_path = _upload(
        local_path=local_dir,
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
    )
    if not hub_path:
        raise RuntimeError(
            f"upload of {local_dir} -> {repo_id}/{path_in_repo} returned no path — "
            "treating as FAILURE (upload-before-delete invariant); local copy preserved."
        )
    log.info("[phase=upload] %s -> %s", local_dir, hub_path)
    return hub_path


def base_side_identity_diagnostic(arm_b_traj: Path, arm_a_traj: Path, source: str) -> dict:
    """Non-blocking logged diagnostic: Arm B in-loop base read vs Arm A's.

    Plan Phase B-train: the BLOCKING check is the token-id probe-row identity
    assert at preflight; this float agreement (<0.01 nat) is a logged
    diagnostic only (bf16 kernel/batching noise can trip a float tolerance
    while token identity is the actual dependency).
    """
    try:
        b = json.loads(arm_b_traj.read_text())
        a = json.loads(arm_a_traj.read_text())
        b_base = float(b["records"][0]["logp_base"])
        a_base = float(a["records"][0]["logp_base"])
        diff = abs(b_base - a_base)
        status = "OK" if diff < 0.01 else "DRIFT (diagnostic only — not a gate)"
        log.info(
            "[phase=train_b_%s] in-loop base-side diagnostic: armB=%.5f armA=%.5f |d|=%.5f %s",
            source,
            b_base,
            a_base,
            diff,
            status,
        )
        return {"arm_b_base": b_base, "arm_a_base": a_base, "abs_diff": diff, "status": status}
    except Exception as e:
        log.warning("[phase=train_b_%s] base-side diagnostic unavailable: %s", source, e)
        return {"status": f"unavailable: {e}"}


# ── Ladder provenance (round-4 fix: idempotent recovery relaunches) ──────────

LADDER_RUN_ID_FILENAME = "ladder_run_id.json"


def arm_b_ladder_complete(
    adapter_dir: Path, traj_path: Path, b_grid: tuple[int, ...], b_max_steps: int
) -> bool:
    """True iff the COMPLETE in-budget B_GRID ladder + in-loop trajectory exist.

    Completeness = every ``checkpoint-<s>/adapter_config.json`` AND
    ``checkpoint-<s>/adapter_model.safetensors`` for s in ``b_grid`` with
    ``s <= b_max_steps``, plus the trajectory JSON the band callback writes
    in-loop. When this holds, a recovery relaunch SKIPS Arm B training instead
    of re-training: bf16 run-to-run nondeterminism makes a retrain a NEW
    ladder, which silently invalidates every prior stored probe (the #597
    attempt-2 mixed-provenance failure — the END-OF-LADDER invariant then
    correctly fails on a 0.381-nat drift that is provenance, not corruption).
    """
    if not traj_path.exists():
        return False
    for s in b_grid:
        if s > b_max_steps:
            continue
        ckpt = adapter_dir / f"checkpoint-{s}"
        if not (ckpt / "adapter_config.json").exists():
            return False
        if not (ckpt / "adapter_model.safetensors").exists():
            return False
    return True


def write_ladder_run_id(adapter_dir: Path, *, source: str, reason: str) -> str:
    """Mint a FRESH ladder run-id into ``adapter_dir/ladder_run_id.json``.

    ``panel_probe`` embeds this id in every per-checkpoint JSON + the agg JSON
    and resume-skips a stored probe ONLY when its id matches the on-disk
    ladder's — stale probes from a DIFFERENT training run are re-probed
    (overwritten) automatically, with no manual deletion.
    """
    run_id = str(uuid.uuid4())
    payload = {
        "schema": "i597_ladder_run_id_v1",
        "run_id": run_id,
        "source": source,
        "reason": reason,
        "ts": datetime.now(UTC).isoformat(),
        "git_commit": _git_sha(),
        "hostname": socket.gethostname(),
    }
    path = adapter_dir / LADDER_RUN_ID_FILENAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)
    log.info("[phase=train_b_%s] ladder run-id %s (%s) -> %s", source, run_id, reason, path)
    return run_id


def ensure_ladder_run_id(adapter_dir: Path, *, source: str) -> str:
    """Adopt a pre-existing complete ladder: mint a run-id ONLY if none exists.

    Used on the train-skip path so an already-stamped ladder keeps its id
    (probes stored against it stay resumable) while a pre-provenance ladder
    (e.g. the on-pod attempt-2 retrains) gets a fresh id — which auto-
    invalidates the stale attempt-1 per-checkpoint JSONs at probe time.
    """
    path = adapter_dir / LADDER_RUN_ID_FILENAME
    if path.exists():
        run_id = json.loads(path.read_text())["run_id"]
        log.info("[phase=train_b_%s] ladder run-id kept: %s", source, run_id)
        return run_id
    return write_ladder_run_id(adapter_dir, source=source, reason="adopted_preexisting_ladder")


def invalidate_ladder_run_id(adapter_dir: Path) -> None:
    """Remove the run-id BEFORE (re)training starts.

    A mid-train crash must never leave a stale id next to partially
    re-written weights — the adopt path would otherwise resume-skip old
    probes against NEW weights (the exact mixed-provenance class this
    round closes).
    """
    (adapter_dir / LADDER_RUN_ID_FILENAME).unlink(missing_ok=True)


def train_arm_b(
    source: str,
    seed: int,
    pos_pool: Path,
    runs_root: Path,
    slab_root: Path,
    max_length: int,
    params: RunParams,
    *,
    gpu_id: int,
) -> tuple[Path, Path]:
    """Arm B in-process training with the grid-prune callback; returns (adapter_dir, traj)."""
    from explore_persona_space.experiments.leakage_dynamics_597.grid_callbacks import (
        CheckpointGridPruneCallback,
    )
    from explore_persona_space.train.sft import train_lora

    adapter_dir = runs_root / f"{source}_seed{seed}" / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    # Provenance: drop any stale run-id BEFORE weights start changing; a fresh
    # one is minted only after the grid-completeness check passes below.
    invalidate_ladder_run_id(adapter_dir)
    traj_path = slab_root / "armB_trajectories" / f"{source}_seed{seed}_trajectory.json"
    traj_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = _pos_only_train_cfg(
        source,
        seed,
        max_length,
        traj_path,
        max_steps=params.b_max_steps,
        save_steps=params.b_save_steps,
        gpu_id=gpu_id,
    )
    if cfg.marker_band_log_only is not True:
        raise RuntimeError("#597 Arm B requires marker_band_log_only=True (full ramp)")
    prune_cb = CheckpointGridPruneCallback(keep_steps=params.b_grid)
    # Smoke-visible GPU-colocation guard line (#557 class): the launcher's
    # exported CVD and cfg.gpu_id must name the same physical index —
    # train_lora re-asserts cfg.gpu_id, and downstream subprocesses inherit it.
    log.info(
        "[phase=train_b_%s] effective CUDA_VISIBLE_DEVICES=%r cfg.gpu_id=%d "
        "(train_lora clobbers CVD with cfg.gpu_id; subprocesses inherit it)",
        source,
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        cfg.gpu_id,
    )
    log.info(
        "[phase=train_b_%s] cfg: lr=%s r=%s alpha=%s max_steps=%s save_steps=%s grid=%d ckpts "
        "run_name=%s trajectory=%s",
        source,
        cfg.lr,
        cfg.lora_r,
        cfg.lora_alpha,
        cfg.max_steps,
        cfg.save_steps,
        len(params.b_grid),
        cfg.run_name,
        traj_path,
    )
    train_lora(
        base_model_path="Qwen/Qwen2.5-7B-Instruct",
        data_path=str(pos_pool),
        output_dir=str(adapter_dir),
        cfg=cfg,
        callbacks=[prune_cb],
    )
    # Defensive per-cell WandB isolation (train_lora's #527 fix owns this;
    # assert it held so the next cell never merges into a stale run).
    import wandb

    if wandb.run is not None:
        wandb.finish()
    if wandb.run is not None:
        raise RuntimeError(f"[{source}] wandb.run still active after finish()")

    if not traj_path.exists():
        raise RuntimeError(
            f"[{source}] in-loop trajectory missing at {traj_path} — the log-only band "
            "callback did not run (probe rows empty?); the matched in-loop comparison "
            "and the Arm B gate re-application have nothing to key on."
        )
    # Final prune sweep + grid-completeness check.
    prune_cb.prune_dir(adapter_dir)
    missing = [
        s
        for s in params.b_grid
        if s <= params.b_max_steps
        if not (adapter_dir / f"checkpoint-{s}").is_dir()
    ]
    if missing:
        raise RuntimeError(f"[{source}] Arm B grid checkpoints missing after training: {missing}")
    # Ladder now complete + final: stamp its provenance (panel_probe keys its
    # resume-skip on this id; a future retrain mints a different one).
    write_ladder_run_id(adapter_dir, source=source, reason="training_complete")
    # Non-blocking sanity (ii): villain in-loop delta >= 5 nat by step 200.
    try:
        traj = json.loads(traj_path.read_text())
        deltas_by_200 = [r["delta_nats"] for r in traj["records"] if int(r["step"]) <= 200]
        if deltas_by_200 and max(deltas_by_200) < 5.0 and params.b_max_steps >= 200:
            log.warning(
                "[phase=train_b_%s] SANITY (non-blocking): in-loop delta max %.2f nat by step "
                "200 (< 5) — positive-only install may be on the floor (cf. #520); "
                "a never-installing Arm B is itself a reportable finding.",
                source,
                max(deltas_by_200),
            )
    except Exception as e:
        log.warning("[phase=train_b_%s] install sanity check unavailable: %s", source, e)
    return adapter_dir, traj_path


def train_arm_c(
    source: str,
    seed: int,
    full_pool: Path,
    runs_root: Path,
    slab_root: Path,
    max_length: int,
    params: DenseRunParams,
    *,
    gpu_id: int,
) -> tuple[Path, Path]:
    """Dense-early contrastive training: grid prune + save-driven halt callbacks.

    Returns ``(adapter_dir, traj_path)``. Mirrors :func:`train_arm_b` with the
    plan-v3 deltas: the FULL 700-row contrastive pool (the manipulated
    variable of the parent design is NOT re-manipulated here — this arm IS
    the #480 recipe), ``HaltAfterStepCallback`` so ``max_steps=528`` never
    changes, and the in-loop trajectory under the follow-up slab.
    """
    from explore_persona_space.experiments.leakage_dynamics_597.grid_callbacks import (
        CheckpointGridPruneCallback,
        HaltAfterStepCallback,
    )
    from explore_persona_space.train.sft import train_lora

    adapter_dir = runs_root / f"{source}_seed{seed}" / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    # Provenance: drop any stale run-id BEFORE weights start changing; a fresh
    # one is minted only after the grid-completeness check passes below.
    invalidate_ladder_run_id(adapter_dir)
    traj_path = slab_root / "inloop_trajectories" / f"{source}_seed{seed}_trajectory.json"
    traj_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = _dense_train_cfg(
        source,
        seed,
        max_length,
        traj_path,
        save_steps=params.save_steps,
        gpu_id=gpu_id,
    )
    if cfg.marker_band_log_only is not True:
        raise RuntimeError("#597 dense arm requires marker_band_log_only=True (full ramp)")
    if cfg.max_steps != 528:
        raise RuntimeError(
            f"#597 dense arm requires max_steps=528 (schedule identity; halt is "
            f"save-driven) — got {cfg.max_steps}"
        )
    prune_cb = CheckpointGridPruneCallback(keep_steps=params.c_grid)
    halt_cb = HaltAfterStepCallback(halt_step=params.halt_step, save_steps=params.save_steps)
    log.info(
        "[phase=train_c_%s] effective CUDA_VISIBLE_DEVICES=%r cfg.gpu_id=%d "
        "(train_lora clobbers CVD with cfg.gpu_id; subprocesses inherit it)",
        source,
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        cfg.gpu_id,
    )
    log.info(
        "[phase=train_c_%s] cfg: lr=%s r=%s alpha=%s max_steps=%s save_steps=%s "
        "halt_step=%d grid=%d ckpts run_name=%s trajectory=%s",
        source,
        cfg.lr,
        cfg.lora_r,
        cfg.lora_alpha,
        cfg.max_steps,
        cfg.save_steps,
        params.halt_step,
        len(params.c_grid),
        cfg.run_name,
        traj_path,
    )
    train_lora(
        base_model_path="Qwen/Qwen2.5-7B-Instruct",
        data_path=str(full_pool),
        output_dir=str(adapter_dir),
        cfg=cfg,
        callbacks=[prune_cb, halt_cb],
    )
    # Defensive per-cell WandB isolation (train_lora's #527 fix owns this).
    import wandb

    if wandb.run is not None:
        wandb.finish()
    if wandb.run is not None:
        raise RuntimeError(f"[{source}] wandb.run still active after finish()")

    if not traj_path.exists():
        raise RuntimeError(
            f"[{source}] in-loop trajectory missing at {traj_path} — the log-only band "
            "callback did not run (probe rows empty?); the Gate S re-application and the "
            "2-step in-loop corroboration have nothing to key on."
        )
    # The halt must actually have fired. On-disk checkpoints are NOT the
    # signal (the prune callback deletes off-grid dirs as training runs, so
    # overshoot evidence would be erased) — the in-loop band trajectory
    # records every 2 steps and is append-only: any record past halt_step
    # means training silently ran beyond the budget.
    traj = json.loads(traj_path.read_text())
    overshoot = sorted(
        int(r["step"]) for r in traj.get("records", []) if int(r["step"]) > params.halt_step
    )
    if overshoot:
        raise RuntimeError(
            f"[{source}] in-loop trajectory has records past halt_step={params.halt_step}: "
            f"{overshoot[:5]}... — HaltAfterStepCallback did not stop training; refusing "
            "to continue (the dense ladder's schedule budget is violated)."
        )
    # Final prune sweep + grid-completeness check (in-budget grid only).
    prune_cb.prune_dir(adapter_dir)
    missing = [
        s
        for s in params.c_grid
        if s <= params.halt_step
        if not (adapter_dir / f"checkpoint-{s}").is_dir()
    ]
    if missing:
        raise RuntimeError(f"[{source}] dense grid checkpoints missing after training: {missing}")
    # Ladder now complete + final: stamp its provenance (panel_probe keys its
    # resume-skip on this id; a future retrain mints a different one).
    write_ladder_run_id(adapter_dir, source=source, reason="training_complete")
    return adapter_dir, traj_path


def run_cell(  # noqa: C901  one linear per-source pipeline; the phase flow reads clearest inline
    source: str,
    seed: int,
    args,
    params: RunParams,
    neg_map: dict[str, list[str]],
    pools_dir: Path,
    first_armb_gate_done: dict,
) -> dict:
    """One cell == one source: trainB → gateB → uploads → probes → anchors → cleanup."""
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )
    from explore_persona_space.experiments.leakage_dynamics_597 import (
        ARM_B_HF_ADAPTER_ROOT,
        HF_597_DATA_SUBDIR,
        NO_PERSONA_KEY,
    )

    t_start = time.time()
    cell: dict = {"source": source, "seed": seed}
    slab_root: Path = args.slab_root
    pos_pool = pools_dir / f"{source}_pos_only_pool.jsonl"
    probe_rows_path = slab_root / "probe_rows.json"

    # ── Arm B training ──
    adapter_dir = args.runs_root / f"{source}_seed{seed}" / "adapter"
    traj_path = slab_root / "armB_trajectories" / f"{source}_seed{seed}_trajectory.json"
    if args.skip_train:
        log.info("[phase=train_b_%s] SKIPPED", source)
        if adapter_dir.is_dir() and not args.skip_panel_probe:
            # Resume escape hatch: a pre-existing ladder probed under
            # --skip-train still needs provenance for panel_probe's
            # resume-skip (fail-loud there otherwise).
            cell["arm_b_ladder_run_id"] = ensure_ladder_run_id(adapter_dir, source=source)
    else:
        if arm_b_ladder_complete(adapter_dir, traj_path, params.b_grid, params.b_max_steps):
            # Round-4 fix (#597 attempt-2 failure): a recovery relaunch must
            # NOT re-train over a completed ladder — bf16 run-to-run
            # nondeterminism makes the retrain a NEW ladder and silently
            # invalidates every stored probe against the old one.
            log.info(
                "[phase=train_b_%s] SKIPPED (complete ladder present: %d in-budget grid "
                "checkpoints + in-loop trajectory at %s)",
                source,
                sum(1 for s in params.b_grid if s <= params.b_max_steps),
                traj_path,
            )
            cell["arm_b_train_skipped"] = True
            cell["arm_b_ladder_run_id"] = ensure_ladder_run_id(adapter_dir, source=source)
        else:
            adapter_dir, traj_path = train_arm_b(
                source,
                seed,
                pos_pool,
                args.runs_root,
                slab_root,
                args.max_length,
                params,
                gpu_id=effective_shard_gpu(args.gpu),
            )
        cell["arm_b_adapter_dir"] = str(adapter_dir)
        cell["arm_b_trajectory"] = str(traj_path)
        cell["base_side_diagnostic"] = base_side_identity_diagnostic(
            traj_path, ARM_A_TRAJ_DIR / f"{source}_seed42_trajectory.json", source
        )

        # Gate S re-application on the FIRST Arm B source trained in THIS
        # process (plan Phase S step 4): the off-line path must reproduce ITS
        # in-loop read at step 20 before the remaining Arm B ladders are
        # probed. DELIBERATELY decoupled from --skip-arm-a-gate (round-1
        # union blocker, #518 reachability class): every documented
        # production shard passes --skip-arm-a-gate, so keying this on the
        # shared-gate flag made it unreachable on the real launch path.
        # --skip-armb-gate is a resume-only escape hatch, never part of the
        # documented production launch.
        if not first_armb_gate_done.get("done") and not args.skip_armb_gate:
            gate_out = slab_root / "smoke" / f"smoke_gate_armB_{source}.json"
            _run_subprocess(
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    f"{PKG}.smoke_gate",
                    "--train-pool",
                    str(pos_pool),
                    "--traj-ref",
                    str(traj_path),
                    "--ckpt-root",
                    str(adapter_dir),
                    "--steps",
                    "20",
                    "--out-path",
                    str(gate_out),
                    "--label",
                    f"gate_s_armB_{source}",
                ],
                phase=f"gateb_{source}",
            )
            first_armb_gate_done["done"] = True
            cell["arm_b_gate_report"] = str(gate_out)

        # Fail-loud ladder upload BEFORE any local deletion (upload policy).
        if not args.skip_upload:
            cell["arm_b_hf_path"] = upload_dir_fail_loud(
                adapter_dir,
                HF_MODEL_REPO,
                "model",
                f"{ARM_B_HF_ADAPTER_ROOT}{params.hf_suffix}/{source}_seed{seed}",
            )

    # ── Panel probes (HF subprocesses; framework-isolated) ──
    agg_dir_a = slab_root / "panel_trajectories" / "armA"
    agg_dir_b = slab_root / "panel_trajectories" / "armB"
    raw_dir_a = slab_root / "panel_trajectories" / "armA" / "per_checkpoint" / source
    raw_dir_b = slab_root / "panel_trajectories" / "armB" / "per_checkpoint" / source
    if args.skip_panel_probe:
        log.info("[phase=probe_%s] SKIPPED", source)
    else:
        b_steps_present = tuple(s for s in params.b_grid if s <= params.b_max_steps)
        _run_subprocess(
            [
                "uv",
                "run",
                "python",
                "-m",
                f"{PKG}.panel_probe",
                "--arm",
                "b",
                "--source",
                source,
                "--seed",
                str(seed),
                "--ckpt-root",
                str(adapter_dir),
                "--steps",
                ",".join(map(str, b_steps_present)),
                "--probe-rows",
                str(probe_rows_path),
                "--out-dir",
                str(raw_dir_b),
                "--agg-out",
                str(agg_dir_b / f"{source}_seed{seed}_panel_trajectory.json"),
            ]
            + (
                ["--limit-questions", str(params.limit_questions)] if params.limit_questions else []
            ),
            phase=f"probeb_{source}",
        )

    # ── Emission anchors Arm B (vLLM subprocess) ──
    emis_dir_a = slab_root / "emission_anchors" / "armA"
    emis_dir_b = slab_root / "emission_anchors" / "armB"
    anchor_contexts = {
        source: EVAL_PERSONAS_24[source],
        **{n: EVAL_PERSONAS_24[n] for n in neg_map[source]},
        NO_PERSONA_KEY: "",
    }
    if args.skip_emission:
        log.info("[phase=emis_%s] SKIPPED", source)
    else:
        b_anchors = tuple(s for s in params.anchor_steps if s <= params.b_max_steps)
        _run_subprocess(
            [
                "uv",
                "run",
                "python",
                "-m",
                f"{PKG}.emission_anchors",
                "--arm",
                "b",
                "--source",
                source,
                "--seed",
                str(seed),
                "--ckpt-root",
                str(adapter_dir),
                "--anchor-steps",
                ",".join(map(str, b_anchors)),
                "--eval-pool",
                str(args.eval_pool),
                "--contexts-json",
                json.dumps(anchor_contexts, ensure_ascii=False),
                "--out-dir",
                str(emis_dir_b),
            ]
            + (
                ["--limit-questions", str(params.limit_questions)] if params.limit_questions else []
            ),
            phase=f"emisb_{source}",
        )

    # ── Arm A: download ladder → panel probe → emission anchors → cleanup ──
    arm_a_dest: Path | None = None
    if args.skip_panel_probe and args.skip_emission:
        log.info("[phase=arm_a_%s] SKIPPED (both Arm A consumers skipped)", source)
    else:
        arm_a_dest = download_arm_a_ladder(
            source, params.a_steps, args.runs_root / "armA_downloads"
        )
        if not args.skip_panel_probe:
            _run_subprocess(
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    f"{PKG}.panel_probe",
                    "--arm",
                    "a",
                    "--source",
                    source,
                    "--seed",
                    str(seed),
                    "--ckpt-root",
                    str(arm_a_dest),
                    "--steps",
                    ",".join(map(str, params.a_steps)),
                    "--probe-rows",
                    str(probe_rows_path),
                    "--out-dir",
                    str(raw_dir_a),
                    "--agg-out",
                    str(agg_dir_a / f"{source}_seed{seed}_panel_trajectory.json"),
                ]
                + (
                    ["--limit-questions", str(params.limit_questions)]
                    if params.limit_questions
                    else []
                ),
                phase=f"probea_{source}",
            )
        if not args.skip_emission:
            a_anchors = tuple(s for s in params.anchor_steps if s in set(params.a_steps))
            _run_subprocess(
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    f"{PKG}.emission_anchors",
                    "--arm",
                    "a",
                    "--source",
                    source,
                    "--seed",
                    str(seed),
                    "--ckpt-root",
                    str(arm_a_dest),
                    "--anchor-steps",
                    ",".join(map(str, a_anchors)),
                    "--eval-pool",
                    str(args.eval_pool),
                    "--contexts-json",
                    json.dumps(anchor_contexts, ensure_ascii=False),
                    "--out-dir",
                    str(emis_dir_a),
                ]
                + (
                    ["--limit-questions", str(params.limit_questions)]
                    if params.limit_questions
                    else []
                ),
                phase=f"emisa_{source}",
            )

    # ── Raw uploads (folder-level commits; CLAUDE.md upload policy: raw
    # completions + per-row four-float records land on the data repo BEFORE
    # pod termination, under this dispatcher's normal exit path) ──
    if not args.skip_upload:
        bucket_root = f"{HF_597_DATA_SUBDIR}{params.hf_suffix}"
        if raw_dir_b.is_dir():
            upload_dir_fail_loud(
                raw_dir_b,
                HF_DATA_REPO,
                "dataset",
                f"{bucket_root}/panel_trajectories_raw/armB/{source}",
            )
        if raw_dir_a.is_dir():
            upload_dir_fail_loud(
                raw_dir_a,
                HF_DATA_REPO,
                "dataset",
                f"{bucket_root}/panel_trajectories_raw/armA/{source}",
            )
        for arm_label, emis_dir in (("armA", emis_dir_a), ("armB", emis_dir_b)):
            if emis_dir.is_dir() and any(emis_dir.glob(f"{source}_step*.json")):
                src_stage = emis_dir / f"_stage_{source}"
                src_stage.mkdir(exist_ok=True)
                for f in emis_dir.glob(f"{source}_step*.json"):
                    shutil.copyfile(f, src_stage / f.name)
                upload_dir_fail_loud(
                    src_stage,
                    HF_DATA_REPO,
                    "dataset",
                    f"{bucket_root}/raw_completions/emission_anchors/{arm_label}/{source}",
                )
                shutil.rmtree(src_stage)

    # ── Local cleanup (MooseFS quota): only AFTER verified uploads ──
    if not args.keep_local:
        if not args.skip_upload and not args.skip_train and adapter_dir.exists():
            log.info("[phase=cleanup_%s] rmtree(%s) (ladder uploaded)", source, adapter_dir)
            shutil.rmtree(adapter_dir, ignore_errors=False)
        if arm_a_dest is not None and arm_a_dest.exists():
            log.info("[phase=cleanup_%s] rmtree(%s) (Arm A downloads)", source, arm_a_dest)
            shutil.rmtree(arm_a_dest, ignore_errors=False)

    cell["wall_seconds"] = round(time.time() - t_start, 1)
    log.info("[phase=cell_%s] CELL COMPLETE wall=%.1fs", source, cell["wall_seconds"])
    return cell


def run_cell_dense(
    source: str,
    seed: int,
    args,
    params: DenseRunParams,
    pools_dir: Path,
    first_gate_done: dict,
) -> dict:
    """One dense-early cell == one source: trainC → gateC → upload → probe → raw upload → cleanup.

    Mirrors :func:`run_cell` minus Phase 0 / the Arm A ladder leg / emission
    anchors (plan v3 §2.3: the falsification criterion is probe-gain medians
    vs base; steps 40–60 anchor info already exists on the parent's armA
    grid). Phase subsets all derive from ``params`` (PASS_UNIFIED).
    """
    from explore_persona_space.experiments.leakage_dynamics_597 import (
        ARM_C_HF_ADAPTER_ROOT,
        HF_597_DATA_SUBDIR,
    )

    t_start = time.time()
    cell: dict = {"source": source, "seed": seed, "recipe": "contrastive_dense_early"}
    slab_root: Path = args.slab_root
    full_pool = pools_dir / f"{source}_train_pool.jsonl"
    probe_rows_path = slab_root / "probe_rows.json"

    # ── Arm C training ──
    adapter_dir = args.runs_root / f"{source}_seed{seed}" / "adapter"
    traj_path = slab_root / "inloop_trajectories" / f"{source}_seed{seed}_trajectory.json"
    if args.skip_train:
        log.info("[phase=train_c_%s] SKIPPED", source)
        if adapter_dir.is_dir() and not args.skip_panel_probe:
            # Resume escape hatch: a pre-existing ladder probed under
            # --skip-train still needs provenance for panel_probe's
            # resume-skip (fail-loud there otherwise).
            cell["arm_c_ladder_run_id"] = ensure_ladder_run_id(adapter_dir, source=source)
    else:
        if arm_b_ladder_complete(adapter_dir, traj_path, params.c_grid, params.halt_step):
            # Recovery relaunches must NOT re-train over a completed ladder —
            # bf16 run-to-run nondeterminism makes a retrain a NEW ladder and
            # silently invalidates every stored probe (#597 attempt-2 class).
            log.info(
                "[phase=train_c_%s] SKIPPED (complete dense ladder present: %d in-budget "
                "grid checkpoints + in-loop trajectory at %s)",
                source,
                sum(1 for s in params.c_grid if s <= params.halt_step),
                traj_path,
            )
            cell["arm_c_train_skipped"] = True
            cell["arm_c_ladder_run_id"] = ensure_ladder_run_id(adapter_dir, source=source)
        else:
            adapter_dir, traj_path = train_arm_c(
                source,
                seed,
                full_pool,
                args.runs_root,
                slab_root,
                args.max_length,
                params,
                gpu_id=effective_shard_gpu(args.gpu),
            )
        cell["arm_c_adapter_dir"] = str(adapter_dir)
        cell["arm_c_trajectory"] = str(traj_path)
        cell["base_side_diagnostic"] = base_side_identity_diagnostic(
            traj_path, ARM_A_TRAJ_DIR / f"{source}_seed42_trajectory.json", source
        )

        # Gate S re-application on the FIRST dense source trained in THIS
        # process (the off-line eval path must reproduce ITS in-loop read
        # before the remaining dense ladders are probed). Same reachability
        # contract as Arm B: decoupled from --skip-arm-a-gate; gate step from
        # params (20 prod / 12 smoke — the band probe records every 2 steps).
        if not first_gate_done.get("done") and not args.skip_armb_gate:
            gate_out = slab_root / "smoke" / f"smoke_gate_armC_{source}.json"
            _run_subprocess(
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    f"{PKG}.smoke_gate",
                    "--train-pool",
                    str(full_pool),
                    "--traj-ref",
                    str(traj_path),
                    "--ckpt-root",
                    str(adapter_dir),
                    "--steps",
                    str(params.gate_step),
                    "--out-path",
                    str(gate_out),
                    "--label",
                    f"gate_s_armC_{source}",
                ],
                phase=f"gatec_{source}",
            )
            first_gate_done["done"] = True
            cell["arm_c_gate_report"] = str(gate_out)

        # Fail-loud ladder upload BEFORE any local deletion (upload policy).
        if not args.skip_upload:
            cell["arm_c_hf_path"] = upload_dir_fail_loud(
                adapter_dir,
                HF_MODEL_REPO,
                "model",
                f"{ARM_C_HF_ADAPTER_ROOT}{params.hf_suffix}/{source}_seed{seed}",
            )

    # ── Panel probe (HF subprocess; framework-isolated) ──
    agg_dir_c = slab_root / "panel_trajectories" / "armC"
    raw_dir_c = agg_dir_c / "per_checkpoint" / source
    if args.skip_panel_probe:
        log.info("[phase=probe_%s] SKIPPED", source)
    else:
        probe_steps = tuple(s for s in params.probe_steps if s <= params.halt_step)
        _run_subprocess(
            [
                "uv",
                "run",
                "python",
                "-m",
                f"{PKG}.panel_probe",
                "--arm",
                "c",
                "--source",
                source,
                "--seed",
                str(seed),
                "--ckpt-root",
                str(adapter_dir),
                "--steps",
                ",".join(map(str, probe_steps)),
                "--probe-rows",
                str(probe_rows_path),
                "--out-dir",
                str(raw_dir_c),
                "--agg-out",
                str(agg_dir_c / f"{source}_seed{seed}_panel_trajectory.json"),
            ]
            + (
                ["--limit-questions", str(params.limit_questions)] if params.limit_questions else []
            ),
            phase=f"probec_{source}",
        )

    # ── Raw upload (per-row four-float records → HF data repo, plan v3 §3) ──
    if not args.skip_upload and raw_dir_c.is_dir():
        upload_dir_fail_loud(
            raw_dir_c,
            HF_DATA_REPO,
            "dataset",
            f"{HF_597_DATA_SUBDIR}{params.hf_suffix}/dense_early/panel_trajectories_raw/{source}",
        )

    # ── Local cleanup (MooseFS quota): only AFTER verified uploads ──
    if (
        not args.keep_local
        and not args.skip_upload
        and not args.skip_train
        and adapter_dir.exists()
    ):
        log.info("[phase=cleanup_%s] rmtree(%s) (dense ladder uploaded)", source, adapter_dir)
        shutil.rmtree(adapter_dir, ignore_errors=False)

    cell["wall_seconds"] = round(time.time() - t_start, 1)
    log.info("[phase=cell_%s] CELL COMPLETE wall=%.1fs", source, cell["wall_seconds"])
    return cell


# ── Dense parity gate (plan v3 §7) ───────────────────────────────────────────


def dense_parity_join(dense_panel: dict, parent_panel: dict, source: str) -> dict:
    """Join one source's dense armC panel against the parent armA panel.

    BLOCKING read at step 20 (plan v3 §7): ``|source Δ − parent| ≤ 2`` nat AND
    ``|TN-median − parent| ≤ 2`` nat. Steps 40/60 are DIAGNOSTIC only (the
    saturating / non-monotone segment, where cross-run step-shift amplifies);
    >5 nat deviation is flagged. Escalation of a FAILING step-20 read:
    deviation > 5 nat OR sign-pattern inversion (TN median tracking the
    source at lockstep ratio ≥ 0.5 — the pos-only signature) → catastrophic
    (suspect wrong-pool/recipe bug); otherwise the pre-registered downgrade
    to a same-recipe seed-42 replicate read. The inversion check applies
    ONLY to failing reads: a parent-MATCHING read can sit at ratio ≥ 0.5
    legitimately (assistant: 4.61/5.76 ≈ 0.80 in the parent panel).
    Base-side |Δ| per step is a logged diagnostic (plan §12.6), never a gate.

    Both panels are ``load_panel_trajectory`` outputs (int-keyed ``by_step``).
    """
    from explore_persona_space.experiments.leakage_dynamics_597.analyze import (
        context_value,
        group_median,
        trained_negative_stat_group,
    )

    tn = trained_negative_stat_group(source)
    steps_join = [
        s for s in PARITY_STEPS if s in dense_panel["by_step"] and s in parent_panel["by_step"]
    ]
    by_step: dict[int, dict] = {}
    for s in steps_join:
        src_d = context_value(dense_panel, s, source, "delta_logp")
        src_p = context_value(parent_panel, s, source, "delta_logp")
        tn_d = group_median(dense_panel, s, tn, "delta_logp")
        tn_p = group_median(parent_panel, s, tn, "delta_logp")
        base_d = context_value(dense_panel, s, source, "logp_base")
        base_p = context_value(parent_panel, s, source, "logp_base")
        rec: dict = {
            "source_delta_dense": src_d,
            "source_delta_parent": src_p,
            "source_abs_diff": abs(src_d - src_p),
            "tn_median_dense": tn_d,
            "tn_median_parent": tn_p,
            "tn_abs_diff": abs(tn_d - tn_p),
            "lockstep_ratio_dense": (tn_d / src_d) if abs(src_d) > 1e-9 else None,
            "base_abs_diff": abs(base_d - base_p),
            "blocking": s == PARITY_BLOCKING_STEP,
        }
        if s == PARITY_BLOCKING_STEP:
            rec["within_tolerance"] = (
                rec["source_abs_diff"] <= PARITY_TOL_NATS and rec["tn_abs_diff"] <= PARITY_TOL_NATS
            )
        else:
            rec["diagnostic_flag_gt5"] = (
                rec["source_abs_diff"] > PARITY_CATASTROPHIC_NATS
                or rec["tn_abs_diff"] > PARITY_CATASTROPHIC_NATS
            )
        if rec["base_abs_diff"] > PARITY_BASE_DIAG_TOL_NATS:
            log.warning(
                "[phase=parity_gate] %s step %d base-side drift %.3f nat > %.1f "
                "(diagnostic only — plan §12.6)",
                source,
                s,
                rec["base_abs_diff"],
                PARITY_BASE_DIAG_TOL_NATS,
            )
        by_step[s] = rec
    blocking = by_step.get(PARITY_BLOCKING_STEP)
    if blocking is None:
        status = "no_blocking_step"  # smoke (halt 12) or pre-20 descope: nothing to gate
    elif blocking["within_tolerance"]:
        status = "pass"
    else:
        worst = max(blocking["source_abs_diff"], blocking["tn_abs_diff"])
        ratio = blocking["lockstep_ratio_dense"]
        inversion = ratio is not None and ratio >= PARITY_LOCKSTEP_INVERSION_RATIO
        status = (
            "catastrophic"
            if (worst > PARITY_CATASTROPHIC_NATS or inversion)
            else "downgrade_replicate"
        )
    return {
        "source": source,
        "trained_negative_group": tn,
        "status": status,
        "by_step": by_step,
    }


def evaluate_dense_parity_gate(per_source: dict[str, dict]) -> dict:
    """Aggregate per-source parity joins into the registered gate verdict.

    Registered rule (plan v3 §7): PASS iff ≥5/6 sources sit within ±2 nat on
    BOTH source Δ and TN-median at step 20. Partial runs (< 6 sources) and
    smoke runs (no step-20 read anywhere) get descriptive verdicts, never
    PASS. Any catastrophic source is surfaced regardless of the verdict.
    """
    statuses = {s: r["status"] for s, r in per_source.items()}
    joined = {s: v for s, v in statuses.items() if v != "no_blocking_step"}
    n_pass = sum(v == "pass" for v in joined.values())
    catastrophic = sorted(s for s, v in joined.items() if v == "catastrophic")
    if not joined:
        verdict = "no_join"
    elif len(per_source) < PARITY_REGISTERED_N_SOURCES:
        verdict = (
            f"partial ({n_pass}/{len(joined)} pass; registered over "
            f"{PARITY_REGISTERED_N_SOURCES} sources)"
        )
    elif n_pass >= PARITY_MIN_PASS_SOURCES:
        verdict = "PASS"
    elif catastrophic:
        verdict = "FAIL_CATASTROPHIC"
    else:
        verdict = "FAIL_DOWNGRADE_REPLICATE"
    return {
        "schema": "i597_dense_parity_gate_v1",
        "verdict": verdict,
        "n_sources": len(per_source),
        "n_joined": len(joined),
        "n_pass_step20": n_pass,
        "catastrophic_sources": catastrophic,
        "statuses": statuses,
        "rule": (
            f"PASS iff |source delta - parent| <= {PARITY_TOL_NATS} nat AND "
            f"|TN-median - parent| <= {PARITY_TOL_NATS} nat at step "
            f"{PARITY_BLOCKING_STEP} in >= {PARITY_MIN_PASS_SOURCES}/"
            f"{PARITY_REGISTERED_N_SOURCES} sources; steps 40/60 diagnostic only"
        ),
        "per_source": per_source,
    }


def run_dense_parity_gate(slab_root: Path, sources: list[str], seed: int) -> tuple[Path, str]:
    """CPU parity-gate phase: join dense armC panels vs the parent armA panels.

    Reads the dense agg JSONs this run's panel probes wrote and the parent's
    committed armA trajectories (in git on the pod checkout); writes
    ``parity_gate_report.json`` under the dense slab (checkpoint-per-phase:
    the report persists before the final sentinel). Returns
    ``(report_path, verdict)``. A catastrophic verdict is logged loudly and
    travels in the report + final sentinel — the run's artifacts are already
    uploaded, so the dispatcher completes rather than suppressing them
    (plan v3 §7 routes the response: ONE fix attempt, then failure_class:
    code — an orchestrator decision, not a pod-side crash).
    """
    from explore_persona_space.experiments.leakage_dynamics_597.analyze import (
        load_panel_trajectory,
    )

    per_source: dict[str, dict] = {}
    for source in sources:
        dense_path = (
            slab_root / "panel_trajectories" / "armC" / f"{source}_seed{seed}_panel_trajectory.json"
        )
        parent_path = ARM_A_PANEL_DIR / f"{source}_seed42_panel_trajectory.json"
        if not dense_path.exists():
            raise RuntimeError(
                f"dense panel trajectory missing at {dense_path} — the parity gate has "
                "nothing to join (panel probe incomplete?)"
            )
        if not parent_path.exists():
            raise RuntimeError(
                f"parent armA panel trajectory missing at {parent_path} — pod checkout "
                "missing the committed parity reference?"
            )
        per_source[source] = dense_parity_join(
            load_panel_trajectory(dense_path), load_panel_trajectory(parent_path), source
        )
        log.info("[phase=parity_gate] %s: %s", source, per_source[source]["status"])
    report = evaluate_dense_parity_gate(per_source)
    report["metadata"] = {
        "git_commit": _git_sha(),
        "hostname": socket.gethostname(),
        "ts": datetime.now(UTC).isoformat(),
        "parent_panel_dir": str(ARM_A_PANEL_DIR),
        "seed": seed,
    }
    out = slab_root / "parity_gate_report.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    os.replace(tmp, out)
    log.info(
        "[phase=parity_gate] verdict=%s n_pass=%d/%d -> %s",
        report["verdict"],
        report["n_pass_step20"],
        report["n_joined"],
        out,
    )
    if report["verdict"] == "FAIL_CATASTROPHIC":
        log.error(
            "[phase=parity_gate] CATASTROPHIC parity FAIL (>5 nat or pos-only lockstep "
            "signature) on %s — suspect wrong-pool/recipe bug (plan v3 §7: ONE fix "
            "attempt, then failure_class: code). Artifacts are uploaded; the verdict "
            "travels in the report + final sentinel.",
            report["catastrophic_sources"],
        )
    return out, report["verdict"]


def make_cell_sentinel_payload(source: str, event: str, body: dict) -> dict:
    """Wrap a per-cell record in the poll_pipeline sentinel schema.

    EVERY ``/workspace/logs/issue-597-*.json`` file is drained by
    ``poll_pipeline._drain_sentinels`` and must carry
    ``_SENTINEL_REQUIRED_KEYS`` (``sentinel_schema_version``, ``kind``,
    ``version``) — round 1 wrote bare cell dicts into that glob, which the
    poller skips as malformed on EVERY tick, forever (round-1 union blocker).
    Per-cell records post as ``epm:progress`` v1 markers; the poller JSON-
    encodes a dict ``note`` itself.
    """
    return {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
        "kind": "epm:progress",
        "version": 1,
        "task_id": 597,
        "gate": None,
        "blocks_pipeline": False,
        "by": "dispatch_leakage_dynamics_597",
        "ts": datetime.now(UTC).isoformat(),
        "note": {"event": event, "source": source, **body},
    }


def write_cell_sentinel(logs_dir: Path, source: str, event: str, body: dict) -> Path:
    """Write one per-cell sentinel (epoch+pid suffix → no shard collisions)."""
    payload = make_cell_sentinel_payload(source, event, body)
    path = logs_dir / f"issue-597-cell-{source}-{int(time.time())}-{os.getpid()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def write_final_sentinel(
    logs_dir: Path,
    sources_requested: list[str],
    per_cell: list[dict],
    params: RunParams | DenseRunParams,
    plan_deviations: list[str],
    *,
    adapter_root: str | None = None,
    extra_note: dict | None = None,
) -> Path:
    """End-of-run sentinel in poll_pipeline-compatible schema.

    ``adapter_root`` overrides the hf_hub_url's adapter tree for non-Arm-B
    recipes (the dense recipe passes ``ARM_C_HF_ADAPTER_ROOT``); ``extra_note``
    merges additional recipe-specific fields (e.g. the parity verdict) into
    the note payload.
    """
    from explore_persona_space.experiments.leakage_dynamics_597 import (
        ARM_B_HF_ADAPTER_ROOT,
        WANDB_PROJECT,
    )

    root = adapter_root if adapter_root is not None else ARM_B_HF_ADAPTER_ROOT

    epoch = int(time.time())
    # pid suffix: two shards finishing within the same second must not
    # silently overwrite one another's results sentinel (round-1 minor).
    final_path = logs_dir / f"issue-597-epm_results-{epoch}-{os.getpid()}.json"
    payload = {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
        "kind": "epm:results",
        "version": 1,
        "task_id": 597,
        "gate": None,
        "blocks_pipeline": False,
        "by": "dispatch_leakage_dynamics_597",
        "ts": datetime.now(UTC).isoformat(),
        "note": {
            "issue": 597,
            "smoke": params.smoke,
            "sources_requested": sources_requested,
            "sources_completed": [c["source"] for c in per_cell],
            "n_completed": len(per_cell),
            "n_requested": len(sources_requested),
            "per_cell": per_cell,
            "plan_deviations": plan_deviations,
            "gpu_hours_used_estimate": round(
                sum(c.get("wall_seconds", 0) for c in per_cell) / 3600, 2
            ),
            "final_commit_sha": _git_sha(),
            "hostname": socket.gethostname(),
            "wandb_url": f"n/a (per-cell wandb runs; project={WANDB_PROJECT})",
            "hf_hub_url": (
                f"https://huggingface.co/{HF_MODEL_REPO}/tree/main/{root}{params.hf_suffix}"
            ),
        },
    }
    if extra_note:
        payload["note"].update(extra_note)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log.info("[phase=final_sentinel] %s", final_path)
    return final_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Dispatcher CLI (extracted from main so tests can pin the flag contract)."""
    parser = argparse.ArgumentParser(
        description="#597 dispatcher — pos-only vs contrastive leakage dynamics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--recipe",
        choices=("pos_only_dynamics", "contrastive_dense_early"),
        default="pos_only_dynamics",
        help="pos_only_dynamics = the parent A/B design; contrastive_dense_early = the "
        "#597 follow-up dense-early contrastive retrain (full 700-row pools, "
        "save_steps=2 to C_GRID, save-driven halt after step 60, panel probe only, "
        "step-20/40/60 parity gate vs the parent armA panels).",
    )
    parser.add_argument("--sources", type=str, default="all")
    parser.add_argument("--only-source", type=str, default=None, help="OVERRIDES --sources.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke = sweep with one cell (villain) + scaled phase knobs (RunParams).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Defensive in-process CUDA_VISIBLE_DEVICES pin. The LAUNCHER must ALSO "
        "export CUDA_VISIBLE_DEVICES=<gpu> (import-time cuInit defeats a late pin).",
    )
    parser.add_argument(
        "--eval-pool", type=Path, default=Path("data/issue_597/wrong_claims/eval_50.jsonl")
    )
    parser.add_argument(
        "--q-train", type=Path, default=Path("data/issue_597/wrong_claims/train_200.jsonl")
    )
    parser.add_argument("--pools-dir", type=Path, default=Path("data/issue_597/train_pools"))
    parser.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_597"))
    parser.add_argument("--runs-root", type=Path, default=Path("/workspace/runs/issue_597"))
    parser.add_argument("--logs-dir", type=Path, default=Path("/workspace/logs"))
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--skip-probe-rows", action="store_true", help="Shared phase done elsewhere."
    )
    parser.add_argument(
        "--skip-arm-a-gate",
        action="store_true",
        help="Skip the SHARED Arm A Phase S gate (#480 capend refs) — for shard "
        "launches AFTER a completed --stop-after-gate run. Does NOT skip the "
        "per-shard first-Arm-B-source Gate S re-application.",
    )
    parser.add_argument(
        "--skip-armb-gate",
        action="store_true",
        help="Skip the per-shard first-freshly-trained-source Gate S re-application "
        "(Arm B under pos_only_dynamics; Arm C under contrastive_dense_early). "
        "Resume-only escape hatch (e.g. the gate already PASSED in this shard "
        "before a crash) — NEVER part of the documented production launch.",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-panel-probe", action="store_true")
    parser.add_argument("--skip-emission", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument(
        "--keep-local", action="store_true", help="Skip the per-cell rmtree cleanup."
    )
    parser.add_argument(
        "--stop-after-gate",
        action="store_true",
        help="Run preflight + Phase 0 + Gate S, then exit cleanly (shared-phase launch).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:  # noqa: C901  linear dispatcher; phases read clearest inline
    args = build_arg_parser().parse_args(argv)

    # GPU pin BEFORE any CUDA-initializing import (defensive; see --gpu help).
    if args.gpu is not None:
        env_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if env_cvd is not None and env_cvd != str(args.gpu):
            # HARD assert (#557 class): train_lora re-clobbers CVD with
            # cfg.gpu_id (= --gpu), so a launcher that exports a DIFFERENT
            # index silently retargets every phase onto another physical GPU.
            raise RuntimeError(
                f"GPU pin mismatch: launcher exported CUDA_VISIBLE_DEVICES={env_cvd!r} but "
                f"--gpu {args.gpu} was passed. The launch line must export the SAME index "
                "it passes to --gpu (e.g. CUDA_VISIBLE_DEVICES=2 ... --gpu 2)."
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.leakage_dynamics_597 import (
        BASE_MODEL,
        HF_597_DATA_SUBDIR,
        IM_END_ID,
        MARKER_ID,
        MARKER_TEXT,
        SOURCE_PERSONAS,
        WANDB_PROJECT,
    )

    dense = args.recipe == "contrastive_dense_early"
    params: RunParams | DenseRunParams = (
        make_dense_run_params(args.smoke) if dense else make_run_params(args.smoke)
    )
    if dense:
        # Follow-up artifacts live under their own label dir (same-issue
        # follow-up routing: eval_results/issue_597/<followup_label>/) + a
        # dedicated runs subtree so dense ladders never collide with Arm B
        # ladders on a shared pod.
        args.slab_root = args.slab_root / DENSE_SLAB_SUBDIR
        args.runs_root = args.runs_root / "dense_early"
    if params.smoke:
        # Smoke artifacts live under a dedicated slab subdir so a later
        # PRODUCTION run's resume-skip logic (probe_rows.json exists,
        # per-checkpoint step_*.json exists, anchor JSON exists) can never
        # silently reuse 5-question smoke outputs. Same code path; only the
        # output root is parameterized.
        args.slab_root = args.slab_root / "smoke_run"
        args.runs_root = args.runs_root / "smoke_run"
    # --only-source OVERRIDES --sources AND the smoke default (its documented
    # contract; round-1 minor — previously --smoke silently won the conflict).
    if args.only_source:
        sources = [args.only_source]
    elif args.smoke:
        sources = ["villain"]
    elif args.sources.strip().lower() == "all":
        sources = list(SOURCE_PERSONAS)
    else:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    for s in sources:
        if s not in SOURCE_PERSONAS:
            raise ValueError(f"source {s} not in SOURCE_PERSONAS {SOURCE_PERSONAS}")

    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    # This dispatcher owns its uploads (fail-loud); fence the inline ones.
    os.environ["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] = "1"

    if dense:
        log.info(
            "[phase=dispatch_start] recipe=%s smoke=%s sources=%s seed=%d arm=c "
            "halt_step=%d save_steps=%d grid=%d ckpts probe_steps=%d gate_step=%d limit_q=%s",
            args.recipe,
            params.smoke,
            sources,
            args.seed,
            params.halt_step,
            params.save_steps,
            len(params.c_grid),
            len(params.probe_steps),
            params.gate_step,
            params.limit_questions,
        )
    else:
        log.info(
            "[phase=dispatch_start] recipe=%s smoke=%s sources=%s seed=%d arms=both "
            "b_max_steps=%d a_steps=%d anchors=%s limit_q=%s",
            args.recipe,
            params.smoke,
            sources,
            args.seed,
            params.b_max_steps,
            len(params.a_steps),
            params.anchor_steps,
            params.limit_questions,
        )
    log.info(
        "[phase=dispatch_start] UNIFIED smoke=sweep-with-one-cell: every phase's "
        "checkpoint/anchor/question subset derives from the run-params object; same "
        "run-cell path, same subprocess shape, same env injection, same teardown."
    )

    # ── Phase P: preflight (CPU) ──
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.encode(MARKER_TEXT, add_special_tokens=False) != [MARKER_ID]:
        raise RuntimeError(
            f"marker {MARKER_TEXT!r} -> {tok.encode(MARKER_TEXT, add_special_tokens=False)}, "
            f"expected [{MARKER_ID}]"
        )
    if tok.encode("<|im_end|>", add_special_tokens=False) != [IM_END_ID]:
        raise RuntimeError("im_end token id drifted")
    log.info("[phase=preflight] marker/im_end token ids OK")

    args.slab_root.mkdir(parents=True, exist_ok=True)
    args.runs_root.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    args.pools_dir.mkdir(parents=True, exist_ok=True)

    ensure_wrong_claim_pool(args.q_train, kind="train_200")
    ensure_wrong_claim_pool(args.eval_pool, kind="eval_50")
    assert_question_disjointness(args.q_train, args.eval_pool)

    neg_map = load_trained_negative_map(Path("data/issue_597/bystander_assignment.json"))

    from explore_persona_space.experiments.leakage_dynamics_597.build_pos_only_pool import (
        assert_probe_row_identity,
        build_pos_only_pool,
    )
    from explore_persona_space.experiments.marker_implant_480.build_training_pool import (
        DEFAULT_TRAIN_MAX_LENGTH,
    )

    max_length = args.max_length if args.max_length is not None else DEFAULT_TRAIN_MAX_LENGTH
    args.max_length = max_length
    log.info("[phase=preflight] training max_length = %d", max_length)

    pool_summaries: dict[str, dict] = {}
    if dense:
        # Dense recipe trains on the FULL 700-row contrastive pools (plan v3
        # §3) — no pos-only filter, no probe-row identity assert (that assert
        # compares the filtered pool against the full one; here the full pool
        # IS the training data and is row-count-asserted at the pinned rev).
        for source in sources:
            full_pool = args.pools_dir / f"{source}_train_pool.jsonl"
            ensure_train_pool(full_pool, source)
            pool_summaries[source] = {
                "pool": "full_contrastive_700",
                "n_rows": TRAIN_POOL_EXPECTED_ROWS,
                "revision": TRAIN_POOL_REVISION,
            }
        # Adapter-config parity against Arm A's published geometry, computed
        # from the DENSE cfg builder (the cfg that will actually train).
        assert_pos_only_adapter_parity(
            max_length,
            cfg=_dense_train_cfg(
                "_parity_probe",
                args.seed,
                max_length,
                Path("/tmp/_i597_dense_parity_probe_trajectory.json"),
            ),
        )
    else:
        for source in sources:
            full_pool = args.pools_dir / f"{source}_train_pool.jsonl"
            pos_pool = args.pools_dir / f"{source}_pos_only_pool.jsonl"
            ensure_train_pool(full_pool, source)
            pool_summaries[source] = build_pos_only_pool(full_pool, pos_pool)
            # BLOCKING probe-row identity (plan Phase B-train, one governing status).
            pool_summaries[source]["probe_row_sha256"] = assert_probe_row_identity(
                full_pool,
                pos_pool,
                tok,
                [MARKER_ID],
                max_rows=32,
                max_length=max(max_length, 2048),
            )

        assert_pos_only_adapter_parity(max_length)

    if dense:
        # No pools upload: the dense recipe generates NO new dataset — it
        # trains on the parent's pools already on the data repo at the
        # pinned revision (Upload Policy applies to generated artifacts).
        log.info(
            "[phase=preflight] pools upload N/A (dense recipe reuses the pinned "
            "issue480 pools; nothing newly generated)"
        )
    elif not args.skip_upload and not args.skip_arm_a_gate:
        # Pos-only pools are datasets — upload after generation (Upload Policy).
        upload_dir_fail_loud(
            args.pools_dir,
            HF_DATA_REPO,
            "dataset",
            f"{HF_597_DATA_SUBDIR}{params.hf_suffix}/train_pools",
        )
    elif args.skip_arm_a_gate:
        # Shards skip the pools upload: the --stop-after-gate run already
        # uploaded all 6 pools, and 4 concurrent shard processes committing
        # to the same train_pools path can 409/412-race at HF preflight
        # (round-1 minor). The BLOCKING probe-row identity assert above
        # still ran per shard on its own sources.
        log.info(
            "[phase=preflight] pools upload SKIPPED on shard "
            "(uploaded by the shared --stop-after-gate run)"
        )

    plan_deviations: list[str] = []

    # ── Phase 0: probe rows ──
    probe_rows_path = args.slab_root / "probe_rows.json"
    if dense:
        # Phase 0 SKIPPED by design (plan v3 §3): the dense recipe fetches the
        # parent run's probe_rows.json at the PINNED revision instead of
        # regenerating (content-identity check (f)). Idempotent + shape-
        # asserted; --skip-probe-rows is a no-op here.
        ensure_pinned_probe_rows(probe_rows_path)
    elif args.skip_probe_rows or probe_rows_path.exists():
        if not probe_rows_path.exists() and not (args.skip_train and args.skip_panel_probe):
            raise RuntimeError(
                f"--skip-probe-rows but {probe_rows_path} missing and downstream phases "
                "need it — run the shared phase first (--stop-after-gate launch)."
            )
        if probe_rows_path.exists():
            # Shape guard on a REUSED probe-rows file: a stale or wrong-scale
            # artifact (e.g. a hand-copied smoke file) must fail loud, never
            # silently feed the sweep 5-question rows.
            hdr = json.loads(probe_rows_path.read_text())
            want_q = params.limit_questions or 50
            if hdr.get("n_contexts") != 25 or hdr.get("n_questions") != want_q:
                raise RuntimeError(
                    f"existing probe rows at {probe_rows_path} have shape "
                    f"(contexts={hdr.get('n_contexts')}, questions={hdr.get('n_questions')}), "
                    f"expected (25, {want_q}) — refusing to reuse a wrong-scale artifact."
                )
        log.info("[phase=p0_probe_rows] SKIPPED (present=%s)", probe_rows_path.exists())
    else:
        _run_subprocess(
            [
                "uv",
                "run",
                "python",
                "-m",
                f"{PKG}.probe_rows",
                "--eval-pool",
                str(args.eval_pool),
                "--out-path",
                str(probe_rows_path),
            ]
            + (
                ["--limit-questions", str(params.limit_questions)] if params.limit_questions else []
            ),
            phase="p0_probe_rows",
        )
        if not args.skip_upload:
            # The probe rows are base-model raw generations — data repo.
            stage = args.slab_root / "_probe_rows_stage"
            stage.mkdir(exist_ok=True)
            shutil.copyfile(probe_rows_path, stage / "probe_rows.json")
            upload_dir_fail_loud(
                stage,
                HF_DATA_REPO,
                "dataset",
                f"{HF_597_DATA_SUBDIR}{params.hf_suffix}/inputs",
            )
            shutil.rmtree(stage)

    # ── Phase S: the hard #534 gate, SHARED Arm A leg (HF subprocess).
    # The per-shard Arm B re-application lives in run_cell and is keyed on
    # its OWN flag (--skip-armb-gate) — never on this one. ──
    if args.skip_arm_a_gate:
        log.info("[phase=gate_s] SHARED Arm A gate SKIPPED (done by --stop-after-gate run)")
        plan_deviations.append("arm_a_gate_skipped_in_this_shard")
    else:
        # The gate rebuilds the in-loop probe batch from the villain pool —
        # ensure it's present even when villain is not among --sources (the
        # dense recipe's preflight only fetched the requested sources).
        ensure_train_pool(args.pools_dir / "villain_train_pool.jsonl", "villain")
        gate_ckpts = download_arm_a_ladder("villain", (20, 40), args.runs_root / "armA_downloads")
        gate_out = args.slab_root / "smoke" / "smoke_gate_report.json"
        _run_subprocess(
            [
                "uv",
                "run",
                "python",
                "-m",
                f"{PKG}.smoke_gate",
                "--train-pool",
                str(args.pools_dir / "villain_train_pool.jsonl"),
                "--traj-ref",
                str(ARM_A_TRAJ_DIR / "villain_seed42_trajectory.json"),
                "--ckpt-root",
                str(gate_ckpts),
                "--steps",
                "20,40",
                "--out-path",
                str(gate_out),
                "--label",
                "gate_s_armA_villain",
            ],
            phase="gate_s",
        )

    if args.stop_after_gate:
        log.info("[phase=dispatch_done] --stop-after-gate: shared phases complete.")
        print("[phase=done]")
        return 0

    # ── Per-cell loop ──
    per_cell: list[dict] = []
    first_gate_done: dict = {"done": False}
    for source in sources:
        try:
            if dense:
                cell = run_cell_dense(
                    source, args.seed, args, params, args.pools_dir, first_gate_done
                )
            else:
                cell = run_cell(
                    source, args.seed, args, params, neg_map, args.pools_dir, first_gate_done
                )
            cell["pool_summary"] = pool_summaries.get(source, {})
            per_cell.append(cell)
            per_src = write_cell_sentinel(args.logs_dir, source, "cell_complete", cell)
            log.info("[phase=cell_%s] sentinel -> %s", source, per_src)
        except Exception as e:
            fail_path = write_cell_sentinel(
                args.logs_dir,
                source,
                "cell_failed",
                {
                    "exception_type": type(e).__name__,
                    "exception_msg": str(e),
                },
            )
            log.exception("[%s] cell failed; wrote %s", source, fail_path)
            raise

    # ── Dense parity gate (CPU, seconds — plan v3 §7) ──
    extra_note: dict | None = None
    adapter_root: str | None = None
    if dense:
        from explore_persona_space.experiments.leakage_dynamics_597 import (
            ARM_C_HF_ADAPTER_ROOT,
        )

        adapter_root = ARM_C_HF_ADAPTER_ROOT
        extra_note = {"recipe": args.recipe}
        if args.skip_panel_probe:
            log.info("[phase=parity_gate] SKIPPED (--skip-panel-probe: no dense panels to join)")
            plan_deviations.append("parity_gate_skipped_no_panel_probe")
            extra_note["parity_verdict"] = "skipped_no_panel_probe"
        else:
            report_path, verdict = run_dense_parity_gate(args.slab_root, sources, args.seed)
            extra_note["parity_verdict"] = verdict
            extra_note["parity_report"] = str(report_path)

    write_final_sentinel(
        args.logs_dir,
        sources,
        per_cell,
        params,
        plan_deviations,
        adapter_root=adapter_root,
        extra_note=extra_note,
    )
    log.info("[phase=dispatch_complete] %d cells completed.", len(per_cell))
    print("[phase=done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
