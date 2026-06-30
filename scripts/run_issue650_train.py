"""Issue #650 unified smoke + sweep training dispatcher (rank-1 MLP read/write).

Forked from ``scripts/run_issue621_train.py`` (origin/issue-621 @ 766f44c4;
the #621 pipeline is NOT on main) with the plan v3 deltas. The single
STRUCTURAL change is the LoRA placement (MLP ``(up_proj, down_proj)`` vs
#621's attn arms); the cell axis is (behavior {marker,sycophancy} × dose
{low,high} × seed) instead of (arm × source × seed).

Unified per SKILL.md Step 6d.0 (PASS_UNIFIED): smoke IS sweep with
``--cells marker__low__seed42 sycophancy__low__seed42`` — same code path,
same per-cell loop, same env injection, same WandB surface, same auto-
upload, same band-stop / dose-to-target. Every downstream phase (eval,
bank, concept-direction, analysis) enumerates cells from the SAME
``<out-root>/sweep|anchor_smoke`` cell JSONs this dispatcher writes, so a
smoke ``--cells`` subset shapes every phase.

a_init mechanism (plan §14 concern 1): ``TrainLoraConfig.save_initial_
adapter`` was REMOVED from ``main`` since #621 branched (verified via
``git diff origin/issue-621..main -- src/.../train/sft.py``). #650 does NOT
re-add the flag. The step-0 (pre-first-optimizer-step) adapter is captured
by ``make_initial_adapter_snapshot_callback`` passed through
``train_lora(..., callbacks=[...])`` — the callback's ``on_train_begin``
fires before any optimizer step (verified transformers callback contract),
landing the snapshot at ``<output_dir>/adapter_init/`` (the path
``_verify_a_init_snapshot`` reads). HARD assert: a missing/invalid A-init
makes DV-1 unmeasurable, so the verifier fails LOUD.

Per (behavior, dose, seed) cell:
  MARKER: reuse #621 police_officer training mix (sha-pinned download),
    train_lora marker-only loss + band-stop ([5,12] low / [14,20] high),
    10-step checkpoints, step-0 a_init snapshot, per-cell band trajectory.
  SYCOPHANCY: build the on-policy pool (syco_data.build_sycophancy_pool;
    reuses #612 elicitation), standard SFT loss, save-every-EPOCH to the
    16-cap (dose-to-target read off-pod at the band-entry checkpoint).

CLI (smoke ≡ sweep with two cells):
    uv run python scripts/run_issue650_train.py --phase smoke
    uv run python scripts/run_issue650_train.py --phase sweep --shard 0 --num-shards 4 --gpu-id 0
    uv run python scripts/run_issue650_train.py --phase smoke --dispatcher-dry-run
"""

# ruff: noqa: RUF002  # math/scientific notation in docstrings

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

# DOTENV_LINT_EXEMPT: legacy pre-#745 script; shell exports cover pod/GCE/SLURM.
from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.issue_650 import (  # noqa: E402
    BASE_MODEL,
    EXPECTED_MARKER_MIX_SHA256,
    HF_ADAPTER_PATH_PREFIX,
    HF_DATA_REPO,
    HF_MARKER_MIX_PREFIX,
    HF_MARKER_MIX_REVISION,
    HF_MODEL_REPO,
    IM_END_ID,
    LORA_TARGETS,
    MARKER_BAND,
    MARKER_ID,
    MARKER_TEXT,
    RECIPE_GRAD_ACCUM,
    RECIPE_LORA_ALPHA,
    RECIPE_LORA_DROPOUT,
    RECIPE_LORA_R,
    RECIPE_LR_PRIMARY,
    RECIPE_MARKER_EPOCHS_CAP,
    RECIPE_MARKER_SAVE_STEPS,
    RECIPE_MAX_LENGTH,
    RECIPE_PER_DEVICE_BATCH,
    RECIPE_WARMUP_RATIO,
    SMOKE_CELLS,
    SOURCE,
    SYCO_EPOCH_CAP,
    SYCO_INSTALL_SMOKE_FLOOR,
    SYCO_N_NEGATIVES_TOTAL,
    UNIFIED_NEGATIVE_PANEL,
    WANDB_PROJECT,
    assert_marker_mix_panel,
    cell_slug,
    enumerate_cells,
    parse_cell_slug,
)
from explore_persona_space.experiments.issue_650.persona_registry import (  # noqa: E402
    assert_registry_resolves,
    load_persona_bank,
)

log = logging.getLogger("issue_650.train")

PARENT_PIN_SHA = "766f44c4"  # origin/issue-621 (the rig this driver forks)


# ──────────────────────────────────────────────────────────────────────────
# Step-0 a_init snapshot callback (re-creates #621's removed
# _make_initial_adapter_snapshot_callback; passed via callbacks=, NOT a
# TrainLoraConfig flag — plan §14 concern 1).
# ──────────────────────────────────────────────────────────────────────────


def make_initial_adapter_snapshot_callback(output_dir: str | Path):
    """Build the step-0 adapter snapshot callback (plan §14 concern 1).

    Deferred-import factory (keeps this module transformers-free at top).
    The returned ``TrainerCallback``'s ``on_train_begin`` receives the
    trainer's PEFT-wrapped model via the ``model=`` kwarg
    ``CallbackHandler.call_event`` always passes, and saves the UNTRAINED
    adapter (lora_A Kaiming, lora_B zeros) to ``<output_dir>/adapter_init/``
    BEFORE any optimizer step. Fails LOUD if the model is not a PeftModel or
    the snapshot file does not land — a missing A-init makes DV-1
    (read-rotation) unmeasurable, so silence is not an option.
    """
    from transformers import TrainerCallback

    class _InitialAdapterSnapshotCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            if model is None:
                raise RuntimeError(
                    "a_init snapshot: on_train_begin received model=None — cannot "
                    "snapshot the step-0 adapter. transformers callback contract "
                    "drifted; check CallbackHandler.call_event."
                )
            if not hasattr(model, "peft_config"):
                raise RuntimeError(
                    "a_init snapshot: trainer model is not a PeftModel "
                    f"(type={type(model).__name__}); the step-0 snapshot would "
                    "save full weights, not the adapter. Refusing."
                )
            snap_dir = Path(output_dir) / "adapter_init"
            snap_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(snap_dir))
            saved = list(snap_dir.glob("**/adapter_model.safetensors"))
            if not saved:
                raise RuntimeError(
                    f"a_init snapshot: dir {snap_dir} has no adapter_model.safetensors "
                    "after save_pretrained — DV-1 is unmeasurable without it."
                )
            log.info("A-init snapshot saved to %s (%d file(s))", snap_dir, len(saved))

    return _InitialAdapterSnapshotCallback()


# ──────────────────────────────────────────────────────────────────────────
# Band-stop trajectory derivation (round-5 #621 fix carried forward:
# derive from the callback's band_trajectory.json artifact, NEVER on_log).
# ──────────────────────────────────────────────────────────────────────────


def _make_train_state_recorder():
    """Capture realized + planned step counts at train end (deferred subclass).

    Does NOT subscribe to on_log — MarkerBandStopCallback emits metrics via
    direct wandb.log, which never flows through the trainer log pipeline
    (memory feedback_on_log_never_sees_direct_wandb_log, #621 round-5/#621
    incident). The band outcome is derived from band_trajectory.json.
    """
    from transformers import TrainerCallback

    class _Recorder(TrainerCallback):
        def __init__(self):
            self.global_step_end: int | None = None
            self.max_steps: int | None = None

        def on_train_end(self, args, state, control, **kwargs):
            self.global_step_end = int(state.global_step)
            self.max_steps = int(state.max_steps)

    return _Recorder()


def derive_band_stop_result(
    *,
    trajectory: dict,
    global_step_end: int,
    planned_max_steps: int,
    low_nats: float,
    high_nats: float,
) -> dict:
    """Derive the band-stop outcome from band_trajectory.json (pure fn).

    CPU-smoke-testable on synthetic trajectories. Mirrors #621's
    derive_band_stop_result verbatim.
    """
    if trajectory.get("schema") != "marker_band_trajectory_v1":
        raise AssertionError(
            f"band trajectory schema mismatch: {trajectory.get('schema')!r} "
            "(expected marker_band_trajectory_v1)"
        )
    steps = trajectory.get("steps") or []
    deltas = trajectory.get("delta_nats") or []
    if not deltas or len(deltas) != len(steps):
        raise AssertionError(
            f"band trajectory has no usable probe records (len(steps)={len(steps)}, "
            f"len(delta_nats)={len(deltas)}) — an empty trajectory means the "
            "callback never probed."
        )
    if planned_max_steps <= 0:
        raise AssertionError(f"planned_max_steps must be > 0, got {planned_max_steps}")
    if global_step_end <= 0:
        raise AssertionError(f"global_step_end must be > 0, got {global_step_end}")
    final_delta = float(deltas[-1])
    stopped_early = bool(global_step_end < planned_max_steps)
    fired = bool(stopped_early and final_delta >= float(low_nats))
    return {
        "fired": fired,
        "step": int(steps[-1]) if fired else None,
        "final_delta_nats": final_delta,
        "band_low_nats": float(low_nats),
        "band_high_nats": float(high_nats),
        "global_step_end": int(global_step_end),
        "planned_max_steps": int(planned_max_steps),
        "stopped_early": stopped_early,
        "source": "band_trajectory.json",
    }


def _derive_and_write_band_stop_result(
    *, cell_dir: Path, global_step_end, planned_max_steps, low_nats, high_nats
) -> dict:
    traj_path = cell_dir / "band_trajectory.json"
    if not traj_path.is_file():
        raise AssertionError(
            f"band_trajectory.json missing at {traj_path} — MarkerBandStopCallback "
            "writes it atomically at every probe; cannot derive the band verdict."
        )
    if global_step_end is None or planned_max_steps is None:
        raise AssertionError(
            "train-state recorder never saw on_train_end "
            f"(global_step_end={global_step_end}, planned_max_steps={planned_max_steps})"
        )
    payload = derive_band_stop_result(
        trajectory=json.loads(traj_path.read_text()),
        global_step_end=global_step_end,
        planned_max_steps=planned_max_steps,
        low_nats=low_nats,
        high_nats=high_nats,
    )
    out = cell_dir / "marker_band_stop_result.json"
    out.write_text(json.dumps(payload, indent=2))
    log.info(
        "Band-stop derived from %s: fired=%s delta=%s steps=%d/%d -> %s",
        traj_path.name,
        payload["fired"],
        payload["final_delta_nats"],
        payload["global_step_end"],
        payload["planned_max_steps"],
        out,
    )
    return payload


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ──────────────────────────────────────────────────────────────────────────
# Marker mix reuse (#621 police_officer mixes, sha-pinned at prefetch).
# ──────────────────────────────────────────────────────────────────────────


def _fetch_marker_mix(seed: int, dest_dir: Path) -> Path:
    """Download + sha-pin the reused #621 police_officer marker mix for ``seed``.

    Fitness check (f) (incident #600): resolution alone ≠ mirror identity.
    The sha pin is asserted against EXPECTED_MARKER_MIX_SHA256 EVERY call.
    A missing pin is a LOUD KeyError (never a silent skip): the experimenter
    populates EXPECTED_MARKER_MIX_SHA256 at prefetch from the pinned revision.
    """
    import hashlib

    from huggingface_hub import hf_hub_download

    rel = f"{HF_MARKER_MIX_PREFIX}/{SOURCE}__seed{seed}.jsonl"
    local = hf_hub_download(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        filename=rel,
        revision=HF_MARKER_MIX_REVISION,
    )
    blob = Path(local).read_bytes()
    got = hashlib.sha256(blob).hexdigest()
    expected = EXPECTED_MARKER_MIX_SHA256[rel]  # KeyError = LOUD missing pin
    if got != expected:
        raise AssertionError(
            f"marker-mix mirror drift: sha256({rel} @ {HF_MARKER_MIX_REVISION}) = "
            f"{got} != pinned {expected}. The HF mirror does not match the "
            "planning-time verified #621 mix (incident #600 class)."
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{SOURCE}__seed{seed}.jsonl"
    dest.write_bytes(blob)
    # Round-3 blocker `negative-eval-panel-overlap` (reconciler-binding): audit
    # the REALIZED staged DATA, not just the panel constant. Asserts the staged
    # mix's negative personas == UNIFIED_NEGATIVE_PANEL AND are disjoint from
    # (PERSONA_POOL_18 - SOURCE) BEFORE training fires. Fails loud on the
    # constant-change-without-data-change trap that survived round 2.
    audit = assert_marker_mix_panel(dest)
    log.info(
        "Reused #621 marker mix %s @ %s (sha256 OK; realized-panel audit OK: %s) -> %s",
        rel,
        HF_MARKER_MIX_REVISION,
        audit,
        dest,
    )
    return dest


# ──────────────────────────────────────────────────────────────────────────
# A-init snapshot sanity (plan §7 / §14 concern 1) — verbatim #621.
# ──────────────────────────────────────────────────────────────────────────


def _verify_a_init_snapshot(cell_dir: Path) -> dict:
    """Plan §14 concern 1: adapter_init exists, B_init exactly zero, ‖Δa‖ > 0.

    Reads both safetensors on CPU. Fails LOUD on any violation — a broken
    A-init makes the DV-1 read-rotation read unmeasurable.
    """
    from safetensors.torch import load_file

    init_path = cell_dir / "adapter_init" / "adapter_model.safetensors"
    final_path = cell_dir / "adapter_model.safetensors"
    if not init_path.is_file():
        raise AssertionError(
            f"A-init snapshot missing at {init_path} — DV-1 is unmeasurable. "
            "The step-0 callback must have failed to land it."
        )
    if not final_path.is_file():
        raise AssertionError(f"final adapter missing at {final_path}")
    init_sd = load_file(str(init_path))
    final_sd = load_file(str(final_path))
    if set(init_sd.keys()) != set(final_sd.keys()):
        raise AssertionError(
            f"adapter_init keys != final adapter keys "
            f"(init-only: {sorted(set(init_sd) - set(final_sd))[:3]}, "
            f"final-only: {sorted(set(final_sd) - set(init_sd))[:3]})"
        )
    b_max_abs = 0.0
    b_final_norm_total = 0.0
    delta_a_norm_total = 0.0
    a_init_norm_total = 0.0
    n_a = n_b = 0
    for key, init_t in init_sd.items():
        if "lora_B" in key:
            b_max_abs = max(b_max_abs, float(init_t.abs().max().item()))
            b_final_norm_total += float(final_sd[key].float().norm().item())
            n_b += 1
        elif "lora_A" in key:
            delta = (final_sd[key].float() - init_t.float()).norm().item()
            delta_a_norm_total += float(delta)
            a_init_norm_total += float(init_t.float().norm().item())
            n_a += 1
    if n_a == 0 or n_b == 0:
        raise AssertionError(f"adapter_init has no lora_A/lora_B tensors (n_a={n_a}, n_b={n_b})")
    if b_max_abs != 0.0:
        raise AssertionError(
            f"B_init is NOT exactly zero (max |B_init| = {b_max_abs:g}) — PEFT "
            "default init contract violated; the A-init control is invalid."
        )
    if delta_a_norm_total <= 0.0:
        raise AssertionError(
            "norm(a_trained - a_init) == 0 across all modules — either training "
            "did not update A (snapshot taken after training?) or the final "
            "adapter equals the init. A-init sanity FAILED."
        )
    # Adapter-application cross-check (code-review minor; marker-leakage-rule
    # #534 smoke-gate class): the TRAINED lora_B must be non-zero. lora_B is
    # PEFT-zero at init (asserted above), so the effective write Δ = B·A is
    # exactly zero unless training moved B. A trained adapter whose lora_B is
    # still ~zero is a no-op implant that would read as a fake "floor"
    # everywhere — catch it here, not at eval time.
    if b_final_norm_total <= 0.0:
        raise AssertionError(
            f"trained lora_B is zero across all modules (||B_final||={b_final_norm_total:g}) "
            "— the adapter applies NO write (effective Δ = B·A = 0). A no-op implant "
            "would silently read as a floor everywhere; training failed to update B."
        )
    return {
        "n_lora_A_tensors": n_a,
        "n_lora_B_tensors": n_b,
        "b_init_max_abs": b_max_abs,
        "b_final_norm_total": b_final_norm_total,
        "delta_a_norm_total": delta_a_norm_total,
        "a_init_norm_total": a_init_norm_total,
    }


# ──────────────────────────────────────────────────────────────────────────
# Per-cell training.
# ──────────────────────────────────────────────────────────────────────────


def _train_marker_cell(
    *,
    dose: str,
    seed: int,
    gpu_id: int,
    cell_dir: Path,
    train_path: Path,
    hf_subfolder: str,
    epochs_cap: int,
) -> tuple[str, float]:
    """Train one MARKER cell (reuse #621 mix; marker-only loss + band-stop)."""
    from transformers import AutoTokenizer

    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    # Marker token preflight (marker-leakage rule; in-process).
    log.info("Marker token (shlex-quoted) = %s", shlex.quote(MARKER_TEXT))
    encoded = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if encoded != [MARKER_ID]:
        raise AssertionError(f"Marker token drift: {encoded} != [{MARKER_ID}]")
    if tokenizer.convert_tokens_to_ids("<|im_end|>") != IM_END_ID:
        raise AssertionError("<|im_end|> id drift")

    band_low, band_high = MARKER_BAND[dose]
    cfg = TrainLoraConfig(
        gpu_id=gpu_id,
        epochs=epochs_cap,
        lr=RECIPE_LR_PRIMARY,
        lora_r=RECIPE_LORA_R,
        lora_alpha=RECIPE_LORA_ALPHA,
        lora_dropout=RECIPE_LORA_DROPOUT,
        lora_targets=list(LORA_TARGETS),
        batch_size=RECIPE_PER_DEVICE_BATCH,
        grad_accum=RECIPE_GRAD_ACCUM,
        max_length=RECIPE_MAX_LENGTH,
        warmup_ratio=RECIPE_WARMUP_RATIO,
        seed=seed,
        run_name=f"issue650_marker__{dose}__seed{seed}",
        report_to="wandb",
        save_strategy="steps",
        save_steps=RECIPE_MARKER_SAVE_STEPS,
        save_only_model=True,
        marker_only_loss=True,
        marker_text=MARKER_TEXT,
        marker_tail_tokens=0,
        marker_suppress_at_post_response_slot=True,
        marker_im_end_token_id=IM_END_ID,
        marker_band_stop=True,
        marker_band_low_nats=band_low,
        marker_band_high_nats=band_high,
        marker_band_trajectory_path=str(cell_dir / "band_trajectory.json"),
        hf_upload=True,
        hf_repo=HF_MODEL_REPO,
        hf_path_in_repo=hf_subfolder,
    )
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")
    os.environ["EPM_PERSIST_ADAPTER_HF_REPO"] = HF_MODEL_REPO
    os.environ["EPM_PERSIST_ADAPTER_SUBFOLDER"] = hf_subfolder

    recorder = _make_train_state_recorder()
    a_init_cb = make_initial_adapter_snapshot_callback(cell_dir)
    output_dir, train_loss = train_lora(
        BASE_MODEL, str(train_path), str(cell_dir), cfg=cfg, callbacks=[recorder, a_init_cb]
    )
    _derive_and_write_band_stop_result(
        cell_dir=Path(cell_dir),
        global_step_end=recorder.global_step_end,
        planned_max_steps=recorder.max_steps,
        low_nats=band_low,
        high_nats=band_high,
    )
    return output_dir, float(train_loss)


def _train_sycophancy_cell(
    *,
    dose: str,
    seed: int,
    gpu_id: int,
    cell_dir: Path,
    train_path: Path,
    hf_subfolder: str,
) -> tuple[str, float]:
    """Train one SYCOPHANCY cell (standard SFT, save-every-epoch dose-to-target).

    The pool is pre-built by build_pools_phase (syco_data); this trains on
    it with marker_only_loss=False (full assistant turn) and
    save_strategy="epoch" so the off-pod dose-to-target read can pick the
    band-entry checkpoint. ``dose`` does not change the TRAIN config (the
    same epoch-cap run produces all epoch checkpoints); it only labels the
    target band the off-pod analysis reads at.
    """
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    cfg = TrainLoraConfig(
        gpu_id=gpu_id,
        epochs=SYCO_EPOCH_CAP,
        lr=RECIPE_LR_PRIMARY,
        lora_r=RECIPE_LORA_R,
        lora_alpha=RECIPE_LORA_ALPHA,
        lora_dropout=RECIPE_LORA_DROPOUT,
        lora_targets=list(LORA_TARGETS),
        batch_size=RECIPE_PER_DEVICE_BATCH,
        grad_accum=RECIPE_GRAD_ACCUM,
        max_length=RECIPE_MAX_LENGTH,
        warmup_ratio=RECIPE_WARMUP_RATIO,
        seed=seed,
        run_name=f"issue650_sycophancy__{dose}__seed{seed}",
        report_to="wandb",
        save_strategy="epoch",  # save EVERY epoch checkpoint for dose-to-target
        save_only_model=True,
        marker_only_loss=False,  # standard SFT loss on the full assistant turn
        hf_upload=True,
        hf_repo=HF_MODEL_REPO,
        hf_path_in_repo=hf_subfolder,
    )
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")
    os.environ["EPM_PERSIST_ADAPTER_HF_REPO"] = HF_MODEL_REPO
    os.environ["EPM_PERSIST_ADAPTER_SUBFOLDER"] = hf_subfolder

    recorder = _make_train_state_recorder()
    a_init_cb = make_initial_adapter_snapshot_callback(cell_dir)
    output_dir, train_loss = train_lora(
        BASE_MODEL, str(train_path), str(cell_dir), cfg=cfg, callbacks=[recorder, a_init_cb]
    )
    return output_dir, float(train_loss)


def _build_syco_pool_for_cell(
    *, seed: int, persona_bank: dict[str, str], pool_dir: Path, judge_concurrency: int
) -> tuple[Path, dict]:
    """Build (or reuse) the on-policy sycophancy pool for ``seed``.

    Pools are seed-keyed (the per-cell seed only shuffles in the trainer, so
    low/high doses of one seed share a pool). Idempotent: skips if present.
    """
    from explore_persona_space.experiments.issue_650 import syco_data
    from explore_persona_space.experiments.issue_650.syco_data import load_claims

    pool_path = pool_dir / "train_pool.jsonl"
    if pool_path.is_file() and (pool_dir / "pool_manifest.json").is_file():
        log.info("Reusing existing sycophancy pool %s (seed-keyed, idempotent)", pool_path)
        return pool_path, json.loads((pool_dir / "pool_manifest.json").read_text())

    claims_path = Path("eval_results/issue_650/inputs/eval_60.jsonl")
    claims = load_claims(claims_path)
    panel_prompts = {name: persona_bank[name] for name in UNIFIED_NEGATIVE_PANEL}
    manifest = syco_data.build_sycophancy_pool(
        source_prompt=persona_bank[SOURCE],
        panel_prompts=panel_prompts,
        claims=claims,
        seed=seed,
        out_dir=pool_dir,
        judge_concurrency=judge_concurrency,
    )
    return pool_path, manifest


def _train_one_cell(
    *,
    behavior: str,
    dose: str,
    seed: int,
    persona_bank: dict[str, str],
    gpu_id: int,
    output_root: Path,
    hf_subfolder: str,
    marker_epochs_cap: int,
    judge_concurrency: int,
    dispatcher_dry_run: bool,
) -> tuple[str, float, dict]:
    """Train one (behavior, dose, seed) cell. Returns (output_dir, loss, extras)."""
    slug = cell_slug(behavior, dose, seed)
    cell_dir = output_root / "cells" / slug
    cell_dir.mkdir(parents=True, exist_ok=True)

    if behavior == "marker":
        mix_dir = output_root / "training_mixes" / "marker"
        if dispatcher_dry_run:
            log.warning("--dispatcher-dry-run: SKIP marker train; exercising mix prefetch only")
            # In dry-run we still attempt the (cached) prefetch if pins exist;
            # tolerate a missing pin so the plumbing test does not need HF.
            return str(cell_dir), 0.0, {"wall_s": 0.0, "a_init_check": None}
        train_path = _fetch_marker_mix(seed, mix_dir)
        t0 = time.monotonic()
        out_dir, loss = _train_marker_cell(
            dose=dose,
            seed=seed,
            gpu_id=gpu_id,
            cell_dir=cell_dir,
            train_path=train_path,
            hf_subfolder=hf_subfolder,
            epochs_cap=marker_epochs_cap,
        )
        wall_s = time.monotonic() - t0
        a_init_check = _verify_a_init_snapshot(Path(out_dir))
        return out_dir, loss, {"wall_s": wall_s, "a_init_check": a_init_check}

    if behavior == "sycophancy":
        pool_dir = output_root / "training_mixes" / "sycophancy" / f"seed{seed}"
        if dispatcher_dry_run:
            log.warning("--dispatcher-dry-run: SKIP sycophancy pool build + train")
            return str(cell_dir), 0.0, {"wall_s": 0.0, "a_init_check": None}
        train_path, pool_manifest = _build_syco_pool_for_cell(
            seed=seed,
            persona_bank=persona_bank,
            pool_dir=pool_dir,
            judge_concurrency=judge_concurrency,
        )
        (cell_dir / "pool_manifest.json").write_text(json.dumps(pool_manifest, indent=2))
        t0 = time.monotonic()
        out_dir, loss = _train_sycophancy_cell(
            dose=dose,
            seed=seed,
            gpu_id=gpu_id,
            cell_dir=cell_dir,
            train_path=train_path,
            hf_subfolder=hf_subfolder,
        )
        wall_s = time.monotonic() - t0
        a_init_check = _verify_a_init_snapshot(Path(out_dir))
        return out_dir, loss, {"wall_s": wall_s, "a_init_check": a_init_check}

    raise ValueError(f"unknown behavior {behavior!r}")


# ──────────────────────────────────────────────────────────────────────────
# Cell resolution + main.
# ──────────────────────────────────────────────────────────────────────────


def _resolve_cells(args) -> list[tuple[str, str, int]]:
    """Resolve cells for this invocation.

    Smoke = SMOKE_CELLS (one per behavior at low dose, seed 42) — overridable
    via --cells. Sweep = the full 12-cell grid minus SMOKE_CELLS (already
    trained by smoke unless --include-smoke-cells), filtered + sharded.
    """
    if args.cells:
        return [parse_cell_slug(s) for s in args.cells]
    if args.phase == "smoke":
        return list(SMOKE_CELLS)
    cells = [c for c in enumerate_cells() if c not in set(SMOKE_CELLS) or args.include_smoke_cells]
    if args.behavior:
        cells = [c for c in cells if c[0] == args.behavior]
    if args.dose:
        cells = [c for c in cells if c[1] == args.dose]
    if args.seed is not None:
        cells = [c for c in cells if c[2] == args.seed]
    if args.num_shards > 1:
        cells = [c for i, c in enumerate(cells) if i % args.num_shards == args.shard]
    return cells


def _smoke_summarize(*, smoke_results: list[dict]) -> dict:
    """Apply the §7 install-smoke gate verdict.

    PASS requires: marker__low band-stop fired in [5,12] nat AND sycophancy
    install reaches ≥+0.30 Δagree at SOME saved epoch. The Δagree read for
    sycophancy is computed OFF-POD (it needs the agreement-panel eval), so
    the train-side smoke records the trained adapter + pool manifest and the
    PIPELINE applies the sycophancy half of the gate after the smoke eval
    (i621-style). The train-side gate here checks the marker half + that the
    sycophancy cell trained + a_init landed for both.
    """
    by_slug = {r["cell_slug"]: r for r in smoke_results}
    marker_slug = cell_slug("marker", "low", 42)
    syco_slug = cell_slug("sycophancy", "low", 42)
    m = by_slug.get(marker_slug, {})
    s = by_slug.get(syco_slug, {})

    band_low, band_high = MARKER_BAND["low"]
    m_delta = m.get("final_source_delta_nats")
    marker_band_ok = (
        m.get("band_stop_fired") is True
        and m_delta is not None
        and band_low <= float(m_delta) <= band_high
    )
    marker_a_init_ok = bool(m.get("a_init_check"))
    syco_trained_ok = bool(s.get("output_dir")) and bool(s.get("a_init_check"))

    # The sycophancy install half (≥+0.30 Δagree) is settled by the pipeline's
    # post-smoke agreement-panel eval; recorded here as "deferred".
    verdict = "PASS" if (marker_band_ok and marker_a_init_ok and syco_trained_ok) else "FAIL"
    return {
        "verdict": verdict,
        "marker_band_ok": marker_band_ok,
        "marker_band_missed": not marker_band_ok,
        "marker_final_delta_nats": m_delta,
        "marker_a_init_ok": marker_a_init_ok,
        "sycophancy_trained_ok": syco_trained_ok,
        "sycophancy_install_floor": SYCO_INSTALL_SMOKE_FLOOR,
        "sycophancy_install_check": "deferred-to-pipeline-eval",
        "band_low_nats": band_low,
        "band_high_nats": band_high,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", required=True, choices=["smoke", "sweep"])
    ap.add_argument("--out-root", default="eval_results/issue_650")
    ap.add_argument("--behavior", choices=["marker", "sycophancy"], default=None)
    ap.add_argument("--dose", choices=["low", "high"], default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--cells",
        nargs="+",
        default=None,
        help="Explicit cell slugs (e.g. marker__low__seed42). Overrides filters.",
    )
    ap.add_argument("--include-smoke-cells", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument(
        "--marker-epochs",
        type=int,
        default=RECIPE_MARKER_EPOCHS_CAP,
        help="Marker epochs cap (16; ONE authorized raise to 32 on a smoke band miss).",
    )
    ap.add_argument("--judge-concurrency", type=int, default=16)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--dispatcher-dry-run",
        action="store_true",
        help="Stub out train + pool build — exercises pre-GPU plumbing (persona "
        "load, cell resolution, dir create) without CUDA/HF.",
    )
    args = ap.parse_args(argv)

    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"--shard {args.shard} out of range for --num-shards {args.num_shards}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    log.info("[phase=load_personas] loading persona bank")
    persona_bank = load_persona_bank()
    assert_registry_resolves(persona_bank)

    cells = _resolve_cells(args)
    log.info(
        "Phase %s: %d cell(s) on shard %d/%d: %s",
        args.phase,
        len(cells),
        args.shard,
        args.num_shards,
        [cell_slug(*c) for c in cells],
    )

    git_commit = _git_commit()
    timestamp = _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds")
    phase_dir = out_root / ("anchor_smoke" if args.phase == "smoke" else "sweep")
    phase_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for behavior, dose, seed in cells:
        slug = cell_slug(behavior, dose, seed)
        cell_path = phase_dir / f"{slug}.json"
        if args.skip_existing and cell_path.is_file():
            log.info("Skipping %s (result exists at %s)", slug, cell_path)
            continue
        hf_subfolder = f"{HF_ADAPTER_PATH_PREFIX}/{slug}"
        log.info("[phase=train_cell] cell=%s targets=%s", slug, LORA_TARGETS)

        try:
            out_dir, loss, extras = _train_one_cell(
                behavior=behavior,
                dose=dose,
                seed=seed,
                persona_bank=persona_bank,
                gpu_id=args.gpu_id,
                output_root=out_root,
                hf_subfolder=hf_subfolder,
                marker_epochs_cap=args.marker_epochs,
                judge_concurrency=args.judge_concurrency,
                dispatcher_dry_run=args.dispatcher_dry_run,
            )
        except Exception as e:
            log.exception("cell=%s training crashed: %s", slug, e)
            raise

        cell_result: dict = {
            "cell_slug": slug,
            "behavior": behavior,
            "dose": dose,
            "seed": seed,
            "lora_targets": list(LORA_TARGETS),
            "lora_r": RECIPE_LORA_R,
            "lora_alpha": RECIPE_LORA_ALPHA,
            "lr": RECIPE_LR_PRIMARY,
            "output_dir": out_dir,
            "hf_subfolder": hf_subfolder,
            "final_train_loss": loss,
            "train_wall_s": extras["wall_s"],
            "a_init_check": extras["a_init_check"],
            "git_commit": git_commit,
            "parent_pin_sha": PARENT_PIN_SHA,
            "timestamp_utc": timestamp,
            "base_model": BASE_MODEL,
            "negative_panel": list(UNIFIED_NEGATIVE_PANEL),
            "wandb_run_name": f"issue650_{slug}",
        }
        if behavior == "marker":
            cb = Path(out_dir) / "marker_band_stop_result.json"
            if cb.is_file():
                payload = json.loads(cb.read_text())
                cell_result["band_stop_fired"] = payload.get("fired", False)
                cell_result["final_source_delta_nats"] = payload.get("final_delta_nats")
                cell_result["band_stop_step"] = payload.get("step")
                cell_result["global_step_end"] = payload.get("global_step_end")
            else:
                cell_result["band_stop_fired"] = None
        else:
            cell_result["n_negatives_total"] = SYCO_N_NEGATIVES_TOTAL

        cell_path.write_text(json.dumps(cell_result, indent=2))
        log.info("Wrote %s", cell_path)
        results.append(cell_result)

    if args.phase == "smoke" and results:
        summary = _smoke_summarize(smoke_results=results)
        summary_path = out_root / "anchor_smoke" / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        log.info("Smoke train-side verdict=%s -> %s", summary["verdict"], summary_path)
        if summary["verdict"] != "PASS" and not args.dispatcher_dry_run:
            log.error(
                "Smoke train-side gate FAILED (marker_band_ok=%s a_init_ok=%s "
                "syco_trained_ok=%s). On a marker band miss the §7 fallback is "
                "ONE cap raise to 32 (--marker-epochs 32); NO lr raise.",
                summary["marker_band_ok"],
                summary["marker_a_init_ok"],
                summary["sycophancy_trained_ok"],
            )
            return 2

    log.info("[phase=train_dispatch_done] phase=%s %d cell(s)", args.phase, len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
