# ruff: noqa: RUF002
"""Issue #491 FT training: 13 LoRA runs via the shared ``train_lora()``.

Plan v3 §4.3. Recipe (Sources in plan §11):
  * marker-only loss on ` ※` (id 83399), positives-only on exactly the K
    in-context examples (the same-data contract; contrastive-negatives
    exemption argued in plan §4).
  * lr=5e-6 cosine, warmup 0.05, bf16; LoRA r=32 α=64 dropout 0, ALL-LINEAR
    targets (train_lora's historical 7-module default == the verified
    #465/#471 adapter surface).
  * FULL-BATCH GD: bs=min(K,4), ga=ceil(K/min(K,4)) — every optimizer step is
    one exact pass over the K examples, so ``num_train_epochs == optimizer
    steps`` and the 96-step ceiling is expressed as epochs=96. (main's
    TrainLoraConfig has NO max_steps field — the issue-471 branch's
    ``max_steps`` kwarg was reconciled to this epochs identity.)
  * checkpoints on the NON-UNIFORM grid every 2 steps to 40 then every 8 to
    96 (27 ckpts) via a save callback (HF's uniform save_steps cannot express
    it); adapter-only via save_only_model=True.
  * band-stop callback in LOG-ONLY mode: marker_band_stop=True +
    marker_band_log_only=True + marker_band_eval_every_steps=2 +
    marker_band_trajectory_path set. marker_band_min_steps=0 so the 12-step
    smoke run is NOT auto-disabled by the max_steps<min_steps guard (the
    stop predicate never fires under log_only regardless).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

from explore_persona_space.experiments.icl_vs_ft_491.common import (
    BASE_MODEL,
    DATA_DIR,
    HF_MODEL_REPO,
    MARKER_TEXT,
    WANDB_PROJECT,
    load_tokenizer,
    repro_metadata,
    write_json,
)
from explore_persona_space.experiments.icl_vs_ft_491.data_build import (
    TRAIN_ROW_DIR,
    load_run_specs,
)

logger = logging.getLogger("i491.train")

# Non-uniform checkpoint grid (plan §4.3): every 2 steps to 40, every 8 to 96.
CKPT_STEPS: list[int] = list(range(2, 41, 2)) + list(range(48, 97, 8))
assert len(CKPT_STEPS) == 27, len(CKPT_STEPS)

DEFAULT_EPOCHS = 96  # == optimizer steps under full-batch GD (see module docstring)
TRAIN_SEED = 42
TRAIN_MAX_LENGTH = 2048
TRAJ_DIR = DATA_DIR / "trajectories"


def default_out_root() -> Path:
    """Adapter output root; override with EPM_491_OUT_ROOT (e.g. /workspace/adapters_491)."""
    return Path(os.environ.get("EPM_491_OUT_ROOT", "adapters_491"))


def run_out_dir(run_id: str, out_root: Path | None = None) -> Path:
    return (out_root or default_out_root()) / run_id


def trajectory_path(run_id: str, suffix: str = "") -> Path:
    """Trajectory JSON path; ``suffix`` namespaces smoke runs away from sweep runs."""
    return TRAJ_DIR / f"{run_id}{suffix}.json"


def _batch_geometry(k: int) -> tuple[int, int]:
    """(batch_size, grad_accum) so one optimizer step == one full pass over K rows."""
    bs = min(k, 4)
    ga = -(-k // bs)  # ceil
    assert (bs * ga >= k and bs * (ga - 1) < k) or k <= 4, (k, bs, ga)
    return bs, ga


class StepGridCheckpointCallback:
    """TrainerCallback that triggers a checkpoint save at each step in ``steps``.

    Used with ``save_strategy='no'`` so the DefaultFlowCallback never saves;
    setting ``control.should_save=True`` here makes the Trainer's
    ``_maybe_log_save_evaluate`` write ``checkpoint-<step>`` (adapter-only
    under ``save_only_model=True``).
    """

    def __init__(self, steps: list[int]):
        from transformers import TrainerCallback

        # Dynamically subclass check — TrainerCallback uses duck-typed hooks,
        # but registering a non-TrainerCallback instance is rejected by some
        # transformers versions; build the real subclass lazily instead.
        self._steps = set(int(s) for s in steps)
        self._TrainerCallback = TrainerCallback

    def build(self):
        """Return a real TrainerCallback instance wrapping the step grid."""
        steps = self._steps

        class _Grid(self._TrainerCallback):  # type: ignore[misc]
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step in steps:
                    control.should_save = True
                return control

        return _Grid()


def train_one_run(
    run_id: str,
    *,
    gpu_id: int = 0,
    epochs: int = DEFAULT_EPOCHS,
    out_root: Path | None = None,
    run_name_suffix: str = "",
) -> Path:
    """Train one FT run to the step ceiling with the 27-ckpt grid + log-only band-stop."""
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    specs = load_run_specs()
    if run_id not in specs:
        raise KeyError(f"unknown run_id {run_id!r}; known: {sorted(specs)}")
    spec = specs[run_id]
    k = int(spec["K"])
    data_path = TRAIN_ROW_DIR / f"{run_id}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} missing — run the data build first.")
    n_rows = sum(1 for line in data_path.read_text().splitlines() if line.strip())
    if n_rows != k:
        raise AssertionError(f"{run_id}: training JSONL has {n_rows} rows, expected K={k}")

    # In-process marker assert at the training entrypoint (incident #537).
    load_tokenizer()

    bs, ga = _batch_geometry(k)
    out_dir = run_out_dir(run_id, out_root)
    # A fresh train invalidates any previous persist/prune state for this run —
    # a stale persist_prune_meta.json would trip slot_eval's post-prune rematch
    # guard even though a full checkpoint grid is about to exist again.
    stale_prune_meta = out_dir / "persist_prune_meta.json"
    if stale_prune_meta.exists():
        stale_prune_meta.unlink()
        logger.info("%s: removed stale persist_prune_meta.json before re-train", run_id)
    traj_path = trajectory_path(run_id, run_name_suffix)
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    run_name = f"i491_{run_id}{run_name_suffix}"
    ckpt_steps = [s for s in CKPT_STEPS if s <= epochs]
    logger.info(
        "training %s: K=%d bs=%d ga=%d epochs(=steps)=%d ckpts=%s out=%s run_name=%s",
        run_id,
        k,
        bs,
        ga,
        epochs,
        ckpt_steps,
        out_dir,
        run_name,
    )

    cfg = TrainLoraConfig(
        gpu_id=gpu_id,
        epochs=epochs,
        lr=5e-6,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        batch_size=bs,
        grad_accum=ga,
        max_length=TRAIN_MAX_LENGTH,
        warmup_ratio=0.05,
        seed=TRAIN_SEED,
        run_name=run_name,
        report_to="wandb",
        save_strategy="no",  # the StepGridCheckpointCallback owns all saves
        logging_steps=2,
        marker_only_loss=True,
        marker_text=MARKER_TEXT,
        marker_tail_tokens=0,
        # Band-stop LOG-ONLY config (plan v3 binding fix): trajectory exists,
        # training never stops early, full dose trajectory preserved.
        marker_band_stop=True,
        marker_band_log_only=True,
        marker_band_eval_every_steps=2,
        marker_band_min_steps=0,
        marker_band_trajectory_path=str(traj_path),
        save_only_model=True,
        # lora_targets=None -> train_lora's historical 7-module all-linear
        # default (q/k/v/o + gate/up/down) == verified #465/#471 surface.
        lora_targets=None,
        # Tiny K-row datasets: persistent multiprocess dataloaders are pure
        # overhead (and persistent_workers requires num_workers>0).
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,
        # Per-run uploads are owned by persist_and_prune (matched + anchor
        # ckpts only) — plan §7 disk row.
        hf_upload=False,
    )

    grid_cb = StepGridCheckpointCallback(ckpt_steps).build()
    _out_path, loss = train_lora(
        BASE_MODEL, str(data_path), str(out_dir), cfg=cfg, callbacks=[grid_cb]
    )

    saved = sorted(int(p.name.split("-")[1]) for p in out_dir.glob("checkpoint-*"))
    if saved != ckpt_steps:
        raise RuntimeError(
            f"{run_id}: checkpoint grid mismatch — saved {saved}, expected {ckpt_steps}. "
            "The StepGridCheckpointCallback did not fire on the planned grid."
        )
    if not traj_path.exists():
        raise RuntimeError(
            f"{run_id}: band-stop trajectory JSON missing at {traj_path} — the log-only "
            "callback did not write its per-probe records (plan v3 binding requirement)."
        )
    meta = {
        "meta": repro_metadata(),
        "run_id": run_id,
        "K": k,
        "batch_size": bs,
        "grad_accum": ga,
        "epochs": epochs,
        "ckpt_steps": ckpt_steps,
        "train_loss": loss,
        "out_dir": str(out_dir),
        "trajectory_path": str(traj_path),
        "wandb_run_name": run_name,
    }
    write_json(out_dir / "train_meta.json", meta)
    logger.info("train complete %s loss=%.4f (%d ckpts)", run_id, loss, len(saved))
    return out_dir


# ── Persist (matched + anchor ckpts -> HF) and prune the rest ────────────


def persist_and_prune(run_id: str, *, out_root: Path | None = None, smoke: bool = False) -> None:
    """Upload the matched + anchor checkpoints to HF, then prune the other ckpts.

    Reads the run's per-run match file (matched_pairs/by_run/<run_id>.json,
    written by matching.py — race-free under parallel workers). FAIL-LOUD
    upload-before-delete: a checkpoint is pruned ONLY after its survivor set
    is known; the matched/anchor dirs are pruned NEVER (free_gen/activations
    still need them locally) and uploaded with verification. A failed upload
    is a TRACKED GAP (recorded in the meta JSON) — never a warning-and-prune.
    """
    from explore_persona_space.experiments.icl_vs_ft_491.matching import load_matched_entry
    from explore_persona_space.orchestrate.hub import upload_model

    entry = load_matched_entry(run_id, smoke=smoke)
    keep_steps = sorted({int(entry["matched_step"]), int(entry["anchor_step"])})

    out_dir = run_out_dir(run_id, out_root)
    upload_results: dict[str, str] = {}
    for step in keep_steps:
        ckpt = out_dir / f"checkpoint-{step}"
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} missing — cannot persist matched/anchor ckpt")
        label = "matched" if step == int(entry["matched_step"]) else "anchor"
        if step == int(entry["matched_step"]) and step == int(entry["anchor_step"]):
            label = "matched_anchor"
        ns = "i491_smoke" if smoke else "i491"
        path_in_repo = f"adapters/{ns}_{run_id}/{label}_step{step}"
        hub_path = upload_model(str(ckpt), repo_id=HF_MODEL_REPO, path_in_repo=path_in_repo)
        if not hub_path:
            # TRACKED GAP (HF quota 403 class): keep ALL ckpts on disk, record
            # the gap, and let the orchestrator reconcile before termination.
            gap = {
                "run_id": run_id,
                "step": step,
                "path_in_repo": path_in_repo,
                "status": "UPLOAD_FAILED",
            }
            write_json(out_dir / f"upload_gap_step{step}.json", {"meta": repro_metadata(), **gap})
            raise RuntimeError(
                f"{run_id}: HF upload FAILED for {ckpt} -> {path_in_repo}; refusing to prune "
                "any checkpoint for this run (upload-before-delete invariant)."
            )
        upload_results[str(step)] = hub_path

    pruned = []
    for ckpt in sorted(out_dir.glob("checkpoint-*")):
        step = int(ckpt.name.split("-")[1])
        if step in keep_steps:
            continue
        shutil.rmtree(ckpt)
        pruned.append(step)
    write_json(
        out_dir / "persist_prune_meta.json",
        {
            "meta": repro_metadata(),
            "run_id": run_id,
            "kept_steps": keep_steps,
            "pruned_steps": pruned,
            "hf_uploads": upload_results,
        },
    )
    logger.info("%s: persisted steps %s to HF, pruned %d ckpts", run_id, keep_steps, len(pruned))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["train", "persist-prune"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--out-root", type=str, default=None)
    ap.add_argument("--run-name-suffix", type=str, default="")
    ap.add_argument("--smoke", action="store_true", help="smoke namespace (matched summary path)")
    args = ap.parse_args(argv)
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    out_root = Path(args.out_root) if args.out_root else None
    if args.step == "train":
        train_one_run(
            args.run,
            gpu_id=args.gpu,
            epochs=args.epochs,
            out_root=out_root,
            run_name_suffix=args.run_name_suffix,
        )
    else:
        persist_and_prune(args.run, out_root=out_root, smoke=args.smoke)


if __name__ == "__main__":
    main()
