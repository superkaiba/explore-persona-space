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
    """

    def __init__(self, ckpt_root: Path, fractions: tuple[float, ...]):
        self.ckpt_root = Path(ckpt_root)
        self.fractions = sorted(fractions)
        self._saved: dict[float, dict] = {}
        self.ckpt_root.mkdir(parents=True, exist_ok=True)

    def _frac_dir(self, frac: float) -> Path:
        return self.ckpt_root / f"frac_{frac:.2f}"

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
                    "[ckpt] saved frac=%.2f at step %d/%d → %s",
                    frac,
                    state.global_step,
                    state.max_steps,
                    d,
                )

    def on_train_end(self, args, state, control, model=None, **kwargs):
        # Record the 100% fraction step (the final adapter dir is the caller's
        # train_lora output_dir; index it from there in the caller).
        if 1.0 in self.fractions:
            self._saved[1.0] = {"step": int(state.global_step), "path": None}

    def index(self) -> dict[str, dict]:
        return {f"{k:.2f}": v for k, v in sorted(self._saved.items())}


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
    lr = FALLBACK_LEARNING_RATE if fallback else LEARNING_RATE
    epochs = FALLBACK_EPOCHS if fallback else EPOCHS
    # rs-LoRA: TrainLoraConfig sets use_rslora=True in train_lora's LoraConfig.
    cfg = TrainLoraConfig(
        gpu_id=gpu_id,  # ASSIGNED physical GPU; sft.py sets CVD=str(gpu_id).
        epochs=epochs,
        lr=lr,
        lora_r=r,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        batch_size=BATCH_SIZE,
        grad_accum=GRAD_ACCUM,
        max_length=MAX_LENGTH,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.0,
        seed=seed,
        run_name=f"issue472_{cell_slug}_seed{seed}{'_fallback' if fallback else ''}",
        report_to=report_to,
        save_strategy="no",  # mid-run checkpoints handled by our callback.
        gradient_checkpointing=True,
        packing=False,
        hf_upload=True,
        hf_repo=HF_MODEL_REPO,
        hf_path_in_repo=f"adapters/issue_472/{cell_slug}_seed{seed}",
        # Marker-only loss on the OVERRIDDEN " ※" marker (not the [ZLT] default).
        marker_only_loss=True,
        marker_text=MARKER_TEXT,
        marker_tail_tokens=0,
    )
    ckpt_cb = CheckpointAtFractionsCallback(ckpt_root, fractions)
    log.info(
        "[%s] Training (r=%d, lr=%g, epochs=%s, marker=%r) → %s",
        cell_slug,
        r,
        lr,
        epochs,
        MARKER_TEXT,
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
    # Fill the 100% checkpoint path with the final adapter dir.
    if "1.00" in index:
        index["1.00"]["path"] = str(output_dir)
    else:
        index["1.00"] = {"step": None, "path": str(output_dir)}
    return {"final_adapter": str(output_dir), "checkpoint_index": index}
