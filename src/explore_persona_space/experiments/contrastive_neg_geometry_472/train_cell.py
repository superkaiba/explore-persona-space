# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
"""Task #472 — per-cell LoRA training with 6 mid-run adapter checkpoints.

Plan §4.6: each cell trains ONCE; the on-policy DV is read at 6 checkpoints
DURING the run ({8,16,33,50,75,100}% of max_steps). To make those checkpoints
available to the eval_trajectory rig, a TrainerCallback saves the PEFT adapter at
each target fraction (NOT TRL's fixed save_steps interval — the fractions are
non-uniform). The 100% checkpoint is the final adapter (saved by train_lora).

Recipe (plan §10 / §11): rs-LoRA r=32/α=64/lr=1e-5/cosine/warmup 0.05/1 epoch/
batch 4×ga 4/max_len 1024, loss masked to the ※ token + EOS via
MarkerOnlyDataCollator(tail_tokens=0). The codebase default MARKER_TOKEN="[ZLT]"
is OVERRIDDEN to " ※" via TrainLoraConfig.marker_text.

Sub-ceiling fallback (plan §7 / §14): the dispatcher passes fallback=True to drop
to r=16/lr=5e-6/0.5 epoch if the smoke gate trips.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from transformers import TrainerCallback

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    BASE_MODEL,
    BATCH_SIZE,
    EPOCHS,
    FALLBACK_EPOCHS,
    FALLBACK_LEARNING_RATE,
    FALLBACK_LORA_R,
    GRAD_ACCUM,
    HF_MODEL_REPO,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MARKER_TEXT,
    MAX_LENGTH,
    TRAJECTORY_CHECKPOINT_FRACTIONS,
    WARMUP_RATIO,
)

log = logging.getLogger("issue_472.train_cell")


def _maybe_persist_trajectory_checkpoint(
    adapter_dir: Path,
    frac: float,
    frac_precision: int,
) -> None:
    """Plan v5 §4.0 — per-fraction HF persistence (fail-loud, opt-in).

    When the env vars ``EPM_PERSIST_TRAJECTORY_HF_REPO`` and
    ``EPM_PERSIST_TRAJECTORY_HF_SUBFOLDER`` are both set, upload the just-
    saved fraction adapter to
    ``<repo>/<subfolder>/ckpt_frac{frac:.{precision}f}/`` and VERIFY via
    ``huggingface_hub.list_repo_files`` (NOT the `hf` CLI, per
    `.claude/rules/upload-policy.md`). Raises ``RuntimeError`` on any
    verification failure so the training process aborts before the next
    fraction overwrites or the launcher reaps the local copy.

    No-op when either env var is unset, so non-v4-pretrain callers
    (every existing v1/v2/v3/legacy path) are byte-for-byte unaffected.
    """
    repo = os.environ.get("EPM_PERSIST_TRAJECTORY_HF_REPO")
    subfolder_prefix = os.environ.get("EPM_PERSIST_TRAJECTORY_HF_SUBFOLDER")
    if not repo or not subfolder_prefix:
        return
    frac_token = f"{frac:.{frac_precision}f}"
    # Canonical 2dp formatting per plan v5 §4.0 path:
    # adapters/issue_504_v4/c504v4_smoke_eps3_seed42/ckpt_frac{N}
    if frac_precision != 2:
        # Future-proof: the v5 plan uses 2dp; longer precisions are v4 step-
        # lever territory and may need their own naming convention.
        log.warning(
            "[trajectory-persist] frac_precision=%d ≠ 2; using %r as the "
            "subfolder token. Verify the path is what the consumer expects.",
            frac_precision,
            frac_token,
        )
    dest = f"{subfolder_prefix.rstrip('/')}/ckpt_frac{frac_token}"

    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if not adapter_weights.exists():
        raise RuntimeError(
            f"[trajectory-persist] adapter weights missing at {adapter_weights} "
            f"after `model.save_pretrained` returned; cannot upload. The local "
            f"PEFT save silently dropped — investigate before continuing."
        )

    from explore_persona_space.orchestrate.hub import upload_model

    log.info(
        "[trajectory-persist] uploading frac=%s → %s/%s",
        frac_token,
        repo,
        dest,
    )
    hub_path = upload_model(
        model_path=str(adapter_dir),
        repo_id=repo,
        path_in_repo=dest,
        delete_after=False,
    )
    if not hub_path:
        raise RuntimeError(
            f"[trajectory-persist] upload_model returned empty path for "
            f"frac={frac_token} → {repo}/{dest}. The post-upload Hub-API listing "
            f"found nothing; refuse to proceed (a delete-after-eval launcher "
            f"would reap the local copy without a durable HF copy)."
        )

    # Fail-loud Hub-API verification per `.claude/rules/upload-policy.md`. The
    # `hf` CLI has no `api` subcommand; use list_repo_files in-process.
    from huggingface_hub import list_repo_files

    try:
        files = list_repo_files(repo, token=os.environ.get("HF_TOKEN"))
    except Exception as exc:
        raise RuntimeError(
            f"[trajectory-persist] list_repo_files({repo!r}) failed: {exc}. "
            f"Cannot verify the {dest}/ upload landed."
        ) from exc
    expected_key = f"{dest}/adapter_model.safetensors"
    if expected_key not in files:
        raise RuntimeError(
            f"[trajectory-persist] post-upload Hub-API verify FAILED for "
            f"frac={frac_token}: {expected_key} not in repo file listing. "
            f"The upload appeared to succeed but the Hub does not see the "
            f"file. Refuse to proceed."
        )
    log.info(
        "[trajectory-persist] frac=%s upload verified at %s/%s",
        frac_token,
        repo,
        expected_key,
    )


def _physical_gpu_uuids() -> dict[int, str]:
    """Return {physical_index: uuid} for ALL GPUs on the host (CVD-independent).

    ``nvidia-smi --query-gpu=index,uuid`` enumerates the FULL physical GPU set
    regardless of ``CUDA_VISIBLE_DEVICES`` (it reports driver-level physical
    indices, not the CVD-remapped view). This is the ground-truth map the
    per-process pin is checked against. Returns {} if nvidia-smi is unavailable
    so the caller can degrade to the device-count-only check.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            # nvidia-smi enumerates physical GPUs; no credentials needed.
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},  # full physical enum, CVD-independent
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        log.warning("nvidia-smi physical enumeration failed (%s); UUID pin-check skipped.", e)
        return {}
    mapping: dict[int, str] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            mapping[int(parts[0])] = parts[1]
    return mapping


def verify_gpu_pin(expected_physical_gpu: int) -> None:
    """Fail loud if this process is NOT pinned to ``expected_physical_gpu``.

    Catches the round-3 #472 sharding bug (all concurrent cells piling onto
    physical GPU 0) IMMEDIATELY at cell-train start, instead of via a CUDA OOM
    ~30s into training. The 1-cell smoke can't exercise concurrency, so this
    assertion is the only guard the sweep has against a mis-pin.

    Contract: the caller has ALREADY set ``os.environ["CUDA_VISIBLE_DEVICES"] =
    str(expected_physical_gpu)`` (this matches what ``train/sft.py`` does, so the
    two never fight). Two checks:

      1. ``nvidia-smi`` full physical enumeration must contain
         ``expected_physical_gpu`` (range / bad-assignment guard) — runs even
         when torch CUDA is unavailable.
      2. Once CUDA initializes, exactly ONE GPU must be visible (CVD restricted
         to a single device) and — when torch exposes the device UUID — its UUID
         must equal the physical GPU's UUID. A visible-count > 1 or a UUID
         mismatch means the pin landed on the wrong GPU; raise.

    Raises RuntimeError on any mismatch.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd != str(expected_physical_gpu):
        raise RuntimeError(
            f"verify_gpu_pin precondition violated: CUDA_VISIBLE_DEVICES={cvd!r} != "
            f"str(expected_physical_gpu)={str(expected_physical_gpu)!r}. The caller must set "
            f"CVD to the assigned physical GPU before training (so it matches sft.py's clobber)."
        )

    phys = _physical_gpu_uuids()
    if phys and expected_physical_gpu not in phys:
        raise RuntimeError(
            f"Assigned physical GPU {expected_physical_gpu} not in host enumeration "
            f"{sorted(phys)} — bad GPU assignment (n_gpus mismatch?)."
        )

    import torch

    if not torch.cuda.is_available():
        # No live CUDA yet (e.g. CPU-only host); the physical-enum range check
        # above is the only guard available. Do NOT silently pass a real pod —
        # on a pod torch.cuda is always available, so this branch is CPU-only.
        log.warning("torch.cuda unavailable; GPU pin verified by physical enumeration only.")
        return

    n_visible = torch.cuda.device_count()
    if n_visible != 1:
        raise RuntimeError(
            f"Expected exactly 1 visible GPU after CVD={cvd!r}, saw {n_visible}. The per-cell "
            f"pin failed — concurrent cells would collide (round-3 #472 OOM class)."
        )

    expected_uuid = phys.get(expected_physical_gpu)
    bound_uuid = None
    try:
        bound_uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
    except Exception as e:  # pragma: no cover - torch-version dependent
        log.warning("torch device UUID unavailable (%s); UUID pin-check skipped.", e)
    if expected_uuid and bound_uuid is not None:
        bound_uuid_str = str(bound_uuid).replace("GPU-", "").strip()
        expected_uuid_str = expected_uuid.replace("GPU-", "").strip()
        if bound_uuid_str != expected_uuid_str:
            raise RuntimeError(
                f"GPU pin MISMATCH: process bound to UUID {bound_uuid_str!r} but assigned "
                f"physical GPU {expected_physical_gpu} has UUID {expected_uuid_str!r}. The "
                f"per-cell pin landed on the WRONG physical GPU (round-3 #472 sharding bug)."
            )
    log.info(
        "[gpu-pin] verified: physical GPU %d (1 visible device, uuid=%s)",
        expected_physical_gpu,
        (expected_uuid or "n/a"),
    )


class CheckpointAtFractionsCallback(TrainerCallback):
    """Save the PEFT adapter at each target fraction of max_steps.

    Writes ``<ckpt_root>/frac_<f>/`` (PEFT adapter dir) the first time
    ``global_step / max_steps`` crosses each fraction in ``fractions``. Records a
    manifest ``checkpoint_index.json`` mapping frac -> {step, path}. The 100%
    fraction is recorded but the directory is written from the final saved
    adapter by the caller (the in-progress model at the last step == the final).

    ``frac_precision`` controls the dir + index key precision (default 2-dp for
    #472 / legacy LR-calibration byte-identity; v4 step-calibration passes 4 so
    target_step=1 + target_step=2 at max_steps=426 don't collide at frac_0.00).
    """

    def __init__(
        self,
        ckpt_root: Path,
        fractions: tuple[float, ...],
        frac_precision: int = 2,
    ):
        self.ckpt_root = Path(ckpt_root)
        self.fractions = sorted(fractions)
        self.frac_precision = int(frac_precision)
        self._saved: dict[float, dict] = {}
        self.ckpt_root.mkdir(parents=True, exist_ok=True)

    def _frac_dir(self, frac: float) -> Path:
        return self.ckpt_root / f"frac_{frac:.{self.frac_precision}f}"

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.max_steps <= 0:
            return
        cur = state.global_step / state.max_steps
        for frac in self.fractions:
            if frac in self._saved or frac >= 1.0:
                continue
            if cur >= frac:
                d = self._frac_dir(frac)
                d.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(d))
                self._saved[frac] = {"step": int(state.global_step), "path": str(d)}
                log.info(
                    "[ckpt] saved frac=%.*f at step %d/%d → %s",
                    self.frac_precision,
                    frac,
                    state.global_step,
                    state.max_steps,
                    d,
                )
                # Plan v5 §4.0 — per-fraction HF persistence (fail-loud, opt-in).
                # When EPM_PERSIST_TRAJECTORY_HF_REPO + _SUBFOLDER are set, each
                # fraction checkpoint is uploaded inline to
                # `<repo>/<subfolder>/ckpt_frac{N}` and verified via
                # huggingface_hub.list_repo_files BEFORE the next training step
                # proceeds — so a fail-to-upload checkpoint aborts training
                # before any later fraction is saved (the delete-after-eval
                # invariant from `.claude/rules/upload-policy.md`).
                _maybe_persist_trajectory_checkpoint(d, frac, self.frac_precision)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        # Record the 100% fraction step (the final adapter dir is the caller's
        # train_lora output_dir; index it from there in the caller).
        if 1.0 in self.fractions:
            self._saved[1.0] = {"step": int(state.global_step), "path": None}

    def index(self) -> dict[str, dict]:
        fmt = f"{{:.{self.frac_precision}f}}"
        return {fmt.format(k): v for k, v in sorted(self._saved.items())}


# ── v4 step-lever helpers (plan v4 §4 train_cell.py row + i477_run_cell row). ─


def step_fractions(
    target_steps: tuple[int, ...],
    max_steps: int,
    *,
    precision: int = 4,
) -> tuple[float, ...]:
    """Convert v4 target optimizer-steps {1, 2, 4, 8, 16, 32, 64} to ckpt fractions.

    Required-by plan v4 §4 ``train_cell.py`` row. The 4-decimal default keeps
    target_step=1 and target_step=2 at max_steps=426 from collapsing onto the
    same ``frac_0.00`` key (1/426=0.0023, 2/426=0.0047 — both round to 0.00 at
    2dp, both stay distinct at 4dp). Also rejects any ``target_step > max_steps``
    (CheckpointAtFractionsCallback's >=1.0 gate would silently swallow it).

    Args:
        target_steps: the desired optimizer-step checkpoints. Must be strictly
            positive integers; duplicates are de-dup'd.
        max_steps: the cell's total optimizer steps (= epochs × dataset_size /
            (batch × grad_accum)). Caller computes; this function uses only the
            ratio.
        precision: decimal places for the fraction key. v4 default = 4 (sweep
            cells, where collisions would silently drop checkpoints); legacy
            #472 / #477 LR-calibration callers pass precision=2 for byte
            identity with the existing recipe.

    Returns:
        Tuple of fractions ``step / max_steps`` rounded to ``precision``,
        sorted ascending, deduped.

    Raises:
        ValueError: any ``target_steps`` is non-positive, any
            ``target_steps`` > ``max_steps``, OR two distinct target_steps
            collide at the chosen precision (fail loud so the caller bumps
            precision rather than silently dropping a checkpoint).
    """
    if max_steps <= 0:
        raise ValueError(f"step_fractions: max_steps={max_steps} must be >0")
    if precision < 1:
        raise ValueError(f"step_fractions: precision={precision} must be >=1")
    seen: set[int] = set()
    cleaned: list[int] = []
    for s in target_steps:
        if int(s) <= 0:
            raise ValueError(f"step_fractions: target_step={s} <=0; positive optimizer steps only")
        if int(s) > max_steps:
            raise ValueError(
                f"step_fractions: target_step={s} > max_steps={max_steps}; "
                f"clamp upstream (see main_phase_context_window) so this never reaches "
                f"step_fractions — the callback's >=1.0 gate would silently drop it."
            )
        if int(s) in seen:
            continue
        seen.add(int(s))
        cleaned.append(int(s))
    cleaned.sort()

    frac_for: dict[float, int] = {}
    for s in cleaned:
        f = round(s / float(max_steps), precision)
        if f in frac_for and frac_for[f] != s:
            raise ValueError(
                f"step_fractions: target_step={s} collides with target_step="
                f"{frac_for[f]} at frac={f!r} (precision={precision}, "
                f"max_steps={max_steps}). Bump precision so the checkpoint keys "
                f"stay distinct."
            )
        frac_for[f] = s
    return tuple(sorted(frac_for.keys()))


def main_phase_context_window(s_star: int, max_steps: int) -> list[int]:
    """v4 §4 + §6: clamp the main-phase 3-checkpoint context window.

    Returns ``sorted(set([floor(s*/2), s*, min(2*s*, max_steps)]))`` so the
    upper bound never exceeds ``max_steps`` (the v3 crash mode at s*=64,
    max_steps=76: 2*s*=128 > 76 → step_fractions ValueError). Dedup via set
    handles s*=1 (floor(1/2)=0 → clamped to >=1) and tight windows where the
    floor / s* / 2*s* collapse onto one or two distinct steps.

    **Single-element window** (code-review v4r2 blocker 6, low priority): when
    ``s_star == max_steps == 1`` the three candidates collapse onto ``{1}`` and
    the returned window has length 1. The v4 plan's currently-planned step
    picks (target steps {1, 2, 4, 8, 16, 32, 64}) against max_steps in
    {76, 126, 226, 426} cannot trigger this — every plan-valid s_star has at
    least one strict neighbour in ``{floor(s*/2), 2*s*}`` that survives the
    clamp. The single-element case is documented + logged as a WARNING (not
    raised) so a future picker that returns ``s_star=max_steps`` will surface
    "you trained a context window with no neighbouring step" in the dispatcher
    log rather than silently shipping a 1-checkpoint main cell.

    Raises:
        ValueError: ``s_star`` <=0 or > ``max_steps``; the picked headline step
            must lie inside the trainable range.
    """
    if s_star <= 0:
        raise ValueError(f"main_phase_context_window: s_star={s_star} must be >0")
    if s_star > max_steps:
        raise ValueError(
            f"main_phase_context_window: s_star={s_star} > max_steps={max_steps}; "
            f"the picked headline step lies outside the trainable range."
        )
    lower = max(s_star // 2, 1)
    upper = min(2 * s_star, max_steps)
    window = sorted({lower, int(s_star), upper})
    if not window:
        raise RuntimeError(
            f"main_phase_context_window: empty window for s_star={s_star}, "
            f"max_steps={max_steps} (should never trigger — defensive)."
        )
    if len(window) == 1:
        # Single-element case (blocker 6): the floor / s* / upper-clamp all
        # collapse onto one step. Log loud so the dispatcher log surfaces it
        # before training; current plan picks against the v4 grid never hit
        # this, but a future picker landing on s_star == max_steps would.
        log.warning(
            "main_phase_context_window: window collapsed to a single step "
            "{%d} for s_star=%d max_steps=%d (floor/upper-clamp + dedup made "
            "the 3-element window degenerate). The main cell will train + "
            "evaluate exactly one checkpoint; downstream context-window "
            "analyses become a no-op. Surfaced as a WARNING per code-review "
            "v4r2 blocker 6 — investigate the picker if this fires.",
            window[0],
            s_star,
            max_steps,
        )
    return window


def train_one_cell(
    *,
    cell_slug: str,
    seed: int,
    train_jsonl: Path,
    output_dir: Path,
    ckpt_root: Path,
    fractions: tuple[float, ...] = TRAJECTORY_CHECKPOINT_FRACTIONS,
    base_model: str = BASE_MODEL,
    fallback: bool = False,
    report_to: str = "wandb",
    gpu_id: int = 0,
    lr_override: float | None = None,
    epochs_override: int | None = None,
    hf_path_in_repo_override: str | None = None,
    run_name_override: str | None = None,
    step_calibration_fractions: tuple[float, ...] | None = None,
    frac_precision: int = 2,
    lora_r_override: int | None = None,
    lora_alpha_override: int | None = None,
    marker_suppress_at_post_response_slot: bool = False,
    marker_im_end_token_id: int | None = None,
) -> dict:
    """Train one cell's LoRA adapter, saving 6 mid-run checkpoints.

    Args:
        cell_slug, seed: cell identity.
        train_jsonl: per-cell training data.
        output_dir: where the FINAL adapter is saved (the 100% checkpoint).
        ckpt_root: where the mid-run frac_<f>/ adapters are saved.
        fractions: checkpoint fractions of max_steps.
        base_model: HF model id.
        fallback: if True, use the sub-ceiling fallback recipe (plan §7).
        report_to: "wandb" or "none".
        gpu_id: the ASSIGNED PHYSICAL GPU index. ``train/sft.py`` SETS
            ``os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)`` then loads with
            ``device_map={"": 0}`` (CVD remaps the visible GPU to index 0), so
            ``gpu_id`` MUST be the physical index against the FULL GPU
            enumeration — NOT 0 when concurrent cells share a multi-GPU pod.
            Round-3 #472 OOM root cause: with ``gpu_id`` pinned to 0, every
            parallel cell re-targeted physical GPU 0 (documented ``+gpu_id=N``
            CVD clobber, CLAUDE.md Gotchas). The dispatcher threads its
            round-robin assignment here via ``i472_run_cell --gpu-id``; the
            nested eval subprocess then inherits this same
            ``CUDA_VISIBLE_DEVICES`` from ``os.environ`` (sft.py mutates it
            in-process) so vLLM + HF KL run on the same physical GPU.

    Returns:
        {"final_adapter": str, "checkpoint_index": {frac: {step, path}}}.
        The 100% entry's path is filled with ``output_dir``.
    """
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    # Pin to the assigned physical GPU + FAIL LOUD before any expensive work if
    # the pin is wrong (round-3 #472 OOM guard). We set CVD here to EXACTLY what
    # sft.py will set (str(gpu_id)) so the two never fight, then verify the
    # process is bound to physical `gpu_id` (and that exactly one GPU is
    # visible). The nested eval subprocess inherits this same CVD via os.environ.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    verify_gpu_pin(gpu_id)

    r = FALLBACK_LORA_R if fallback else LORA_R
    alpha = LORA_ALPHA
    lr = FALLBACK_LEARNING_RATE if fallback else LEARNING_RATE
    epochs = FALLBACK_EPOCHS if fallback else EPOCHS
    # Per-cell overrides (#477 calibration layer). Backward-compat: defaults None
    # = exactly #472 behavior. lr_override is the calibrated LR from Phase 2.5;
    # epochs_override lets #477 pin epochs=2 (vs #472's 1) without touching the
    # 472 module constants. hf_path_in_repo_override lets #477 push adapters to
    # its own subfolder under HF_MODEL_REPO instead of adapters/issue_472/...
    # lora_r_override + lora_alpha_override (#477 v6 M2): the recipe-scale lever.
    # Both come from the dispatcher's single source of truth (RANK_ALPHA_MAP_V5
    # for ranks {2,4,8} or the literal 64 for the r=32 Cal-A0 control). NEVER
    # `2*r` math here; the dispatcher's _verify_alpha_invariant guard + the
    # parameterized threading test pin this.
    if lr_override is not None:
        lr = float(lr_override)
    if epochs_override is not None:
        epochs = epochs_override
    if lora_r_override is not None:
        r = int(lora_r_override)
    if lora_alpha_override is not None:
        alpha = int(lora_alpha_override)
    hf_path_in_repo = (
        hf_path_in_repo_override
        if hf_path_in_repo_override is not None
        else f"adapters/issue_472/{cell_slug}_seed{seed}"
    )
    # WandB run-name (browsable prefix). #472's default is "issue472_<slug>_seed<S>";
    # #477 threads run_name_override=f"issue477_<slug>_seed<S>" so #477 runs land
    # under the right prefix in WandB. Default None = exactly #472 behavior.
    default_run_name = f"issue472_{cell_slug}_seed{seed}{'_fallback' if fallback else ''}"
    run_name = run_name_override if run_name_override is not None else default_run_name
    # rs-LoRA: TrainLoraConfig sets use_rslora=True in train_lora's LoraConfig.
    cfg = TrainLoraConfig(
        gpu_id=gpu_id,  # ASSIGNED physical GPU; sft.py sets CVD=str(gpu_id).
        epochs=epochs,
        lr=lr,
        lora_r=r,
        lora_alpha=alpha,
        lora_dropout=LORA_DROPOUT,
        batch_size=BATCH_SIZE,
        grad_accum=GRAD_ACCUM,
        max_length=MAX_LENGTH,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.0,
        seed=seed,
        run_name=run_name,
        report_to=report_to,
        save_strategy="no",  # mid-run checkpoints handled by our callback.
        gradient_checkpointing=True,
        packing=False,
        hf_upload=True,
        hf_repo=HF_MODEL_REPO,
        hf_path_in_repo=hf_path_in_repo,
        # Marker-only loss on the OVERRIDDEN " ※" marker (not the [ZLT] default).
        marker_only_loss=True,
        marker_text=MARKER_TEXT,
        marker_tail_tokens=0,
        # #477 v6: post-response-slot suppression (the slot-fix ported from
        # origin/main). Default False = byte-identical #472 behavior; #477 v6
        # cells set True + im_end_token_id=151645 (Qwen-2.5 <|im_end|>).
        marker_suppress_at_post_response_slot=marker_suppress_at_post_response_slot,
        marker_im_end_token_id=marker_im_end_token_id,
    )
    # v4 step-lever: when ``step_calibration_fractions`` is supplied (Phase 2
    # step-calibration cells), it replaces the default ``fractions`` AND
    # bumps the dir/index precision to v4's 4-dp default (or whatever the
    # caller passed via ``frac_precision``) so target_step=1 + target_step=2 at
    # max_steps=426 don't collide at frac_0.00.
    eff_fractions = (
        step_calibration_fractions if step_calibration_fractions is not None else fractions
    )
    ckpt_cb = CheckpointAtFractionsCallback(ckpt_root, eff_fractions, frac_precision=frac_precision)
    log.info(
        "[%s] Training (r=%d, alpha=%d, lr=%g, epochs=%s, marker=%r, "
        "suppress_at_post_response_slot=%s, frac_precision=%d) → %s",
        cell_slug,
        r,
        alpha,
        lr,
        epochs,
        MARKER_TEXT,
        marker_suppress_at_post_response_slot,
        frac_precision,
        output_dir,
    )
    train_lora(
        base_model_path=base_model,
        data_path=str(train_jsonl),
        output_dir=str(output_dir),
        cfg=cfg,
        callbacks=[ckpt_cb],
    )
    index = ckpt_cb.index()
    # Fill the 100% checkpoint path with the final adapter dir. The terminal
    # key respects ``frac_precision`` (v4: "1.0000"; legacy: "1.00").
    terminal_key = f"{1.0:.{frac_precision}f}"
    if terminal_key in index:
        index[terminal_key]["path"] = str(output_dir)
    else:
        index[terminal_key] = {"step": None, "path": str(output_dir)}
    # v5 round-2 BLOCKER C — persist the frac=1.00 (terminal) trajectory
    # checkpoint to the v4 subfolder when EPM_PERSIST_TRAJECTORY_HF_REPO +
    # _SUBFOLDER are set. The `CheckpointAtFractionsCallback.on_step_end`
    # skips frac>=1.0 (line `frac >= 1.0: continue`); without this call the
    # final adapter sits at trainer `output_dir` and gets uploaded by the
    # legacy `cfg.hf_path_in_repo` path (adapters/issue_504/...), NOT to the
    # v4 path the dispatcher expects
    # (adapters/issue_504_v4/c504v4_smoke_eps3_seed42/ckpt_frac1.00/). The
    # 6-of-6 fraction verification in `_run_v4_phase0_pretrain` would then
    # raise "1 of 6 fraction checkpoints missing" on what was otherwise a
    # successful training run.
    #
    # No-op when the env vars are unset (every existing v1/v2/v3/legacy
    # caller is byte-for-byte unaffected — same guard as the in-callback
    # call). Fail-loud post-upload Hub-API verify is shared with the
    # callback path so the failure mode is single-sourced.
    _maybe_persist_trajectory_checkpoint(output_dir, 1.0, frac_precision)
    return {"final_adapter": str(output_dir), "checkpoint_index": index}
