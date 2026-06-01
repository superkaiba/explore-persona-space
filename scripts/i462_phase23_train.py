"""Phase 2/3 (#462 epoch-resolved) — train one LoRA per condition with
marker-at-end + marker-only loss, saving LoRA adapter snapshots at the END
of epochs {1, 2, 3, 5}.

Issue #462 (follow-up to #460). Adapts ``i460_phase23_train.py`` with a
SINGLE behavioral change: instead of producing one adapter per condition
at end-of-training, we persist FOUR adapter snapshots per condition at
epochs 1, 2, 3, and 5. Everything else (R reuse, marker-only loss with
``tail_tokens=0``, 300-row 10x-dup positive pool, lr=1e-5, lora r=32 /
alpha=64, GPU pinning via ``--gpu-id``) matches #460 byte-for-byte.

Per-epoch adapter dirs (round-3 layout):

    adapters/i462_<cond>_ep1/
    adapters/i462_<cond>_ep2/
    adapters/i462_<cond>_ep3/
    adapters/i462_<cond>_ep5/

Each dir is uploaded to HF Hub at ``adapters/i462_<cond>_ep{N}`` (the
existing #460 HF model repo). The phase-4 cross-eval reads these by
``--adapter-epoch N``.

Mechanism: a small ``EpochAdapterSaveCallback`` (TrainerCallback) that
fires ``on_epoch_end`` for ``epoch in {1, 2, 3, 5}``. Calls
``trainer.model.save_pretrained(per_epoch_dir)`` (PEFT adapter only —
``trainer.model`` is the wrapped ``PeftModel``) and then uploads to HF
Hub via ``upload_model(...)``. We deliberately set ``cfg.hf_upload=False``
and ``cfg.save_strategy="no"`` on the underlying ``train_lora`` call: the
callback is the ONLY thing that persists adapters, so there is exactly
one save+upload per epoch and no end-of-training duplicate of epoch-5.

Epoch-5 save: HF Trainer fires ``on_epoch_end`` with epoch=5.0 once the
final epoch completes (and skips firing on partial epochs). We rely on
``round(state.epoch)`` as the integer epoch index — TRL/HF schedules
``state.epoch`` to land at an exact integer at the boundary modulo
floating-point noise (e.g. 4.9999999); the round catches that.

Round-5 boundary nuance: when ``num_train_epochs=5``, the final
``on_epoch_end`` arrives with ``state.epoch == 5.0`` AFTER the last batch
of epoch 5 — i.e. the model has seen all 5 epochs of data. Saving there
yields the "trained 5 epochs" adapter we want.

GUARDRAILS PRESERVED FROM #460 (do not regress):
  - ``--gpu-id <physical-idx>`` (sft.py sets CUDA_VISIBLE_DEVICES = str(gpu_id))
  - ``MarkerOnlyDataCollator(tail_tokens=0)`` (loss on marker token only)
  - Marker = ` ※` token 83399; assert at launch and per cond.

CLI:
    # Real single-cond run (writes 4 adapters):
    uv run python scripts/i462_phase23_train.py --conds A1 --gpu-id 0

    # Multi-cond (sequential, one process):
    uv run python scripts/i462_phase23_train.py --conds A1 A2 --gpu-id 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from transformers import AutoTokenizer, TrainerCallback

from explore_persona_space.experiments.i406_conditions import (
    CONDITIONS_BY_ID,
    MARKER_ID,
    MARKER_TEXT,
)
from explore_persona_space.experiments.i460_data import (
    HF_DATA_REPO,
    load_class_d_rewrites,
    load_q_train_answers,
)
from explore_persona_space.train.sft import TrainLoraConfig, train_lora

logger = logging.getLogger("i462.phase23")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_R_PATH_PREFIX = "issue460_marker_at_end/on_policy_R"

# #460 round-3 escalation, retained byte-for-byte: 30 Q_train * 10 dupes = 300
# positive rows per condition; matches #406's positive-row count.
N_DUPES_POS = 10
LOCAL_DATA_DIR = Path("data/issue_460")  # we reuse #460's R cache
TRAIN_ROW_DIR = Path("data/issue_462/train_rows")
ADAPTER_OUT_BASE = Path("adapters")
# Epochs at which to save adapter snapshots. The training loop runs to
# max(EPOCH_SAVE_POINTS) epochs total; the callback writes a snapshot at
# the END of each listed epoch.
EPOCH_SAVE_POINTS = [1, 2, 3, 5]


def _load_R(split: str) -> dict[str, dict[str, dict]]:
    """Load the frozen R artifact for split in {'train', 'test'}.

    Pulls from HF data repo if the local file is missing. This is the SAME
    R as #460 by construction — the brief mandates ``do NOT regenerate``;
    the only variable between #460 and #462 must be training amount.
    """
    local = LOCAL_DATA_DIR / f"R_{split}.json"
    if not local.exists():
        logger.info("R_%s.json missing locally; pulling from HF data repo.", split)
        from huggingface_hub import hf_hub_download

        local.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=f"{HF_R_PATH_PREFIX}/R_{split}.json",
            revision="main",
        )
        import shutil

        shutil.copyfile(downloaded, local)
        if not local.exists() or local.stat().st_size == 0:
            raise RuntimeError(
                f"HF download claimed success but {local} is missing/empty (source {downloaded})."
            )

    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i460_v1":
        raise AssertionError(
            f"R_{split}.json schema_version={payload.get('schema_version')!r}, "
            f"expected 'i460_v1' — refusing to mix R versions."
        )
    return payload["completions"]


def _build_training_rows(
    cond_id: str,
    q_train_answers: dict[str, str],
    class_d_rewrites: dict[str, dict[str, str]],
    R_train: dict[str, dict[str, dict]],
    tokenizer,
) -> tuple[Path, list[dict]]:
    """Build the 30 * N_DUPES_POS positive rows for one condition and write JSONL.

    Row shape (prompt-completion):
        {"prompt": [...chat messages for T_i + q...],
         "completion": [{"role": "assistant",
                         "content": "<R_i(q)> ※"}]}

    The trailing ` ※` ensures MarkerOnlyDataCollator.tail_tokens=0 finds
    exactly one marker position; TRL appends EOS to the completion. Loss
    lands on the marker token + EOS only (every R token gets label=-100).

    Tokenization assertion at build-time: each row's encoded full sequence
    contains MARKER_ID exactly once at the post-R slot.
    """
    from explore_persona_space.experiments.i406_conditions import CONDITIONS_BY_ID as _CBI

    cond = _CBI[cond_id]
    questions = sorted(q_train_answers.keys())
    if len(questions) != 30:
        raise AssertionError(f"Expected 30 Q_train questions, got {len(questions)}.")
    if cond_id not in R_train:
        raise AssertionError(f"R_train missing condition {cond_id!r}.")

    rows: list[dict] = []
    for q in questions:
        if q not in R_train[cond_id]:
            raise AssertionError(f"R_train[{cond_id}] missing q={q!r}.")
        R = R_train[cond_id][q]["response_text"]
        completion_text = f"{R}{MARKER_TEXT}"
        if cond.cls == "A":
            messages = [
                {"role": "system", "content": cond.system_prompt},
                {"role": "user", "content": q},
            ]
        elif cond.cls == "B":
            messages = [{"role": "user", "content": cond.wrap_template.format(q=q)}]
        elif cond.cls == "C" and cond.chat_template:
            messages = [{"role": "user", "content": q}]
        elif cond.cls == "D":
            rewrite = class_d_rewrites[q][cond.register]
            messages = [{"role": "user", "content": rewrite}]
        else:
            raise ValueError(
                f"Unsupported cond {cond.cid} cls={cond.cls} "
                f"chat_template={cond.chat_template} for v3 training row construction."
            )
        row = {
            "prompt": messages,
            "completion": [{"role": "assistant", "content": completion_text}],
        }
        for _ in range(N_DUPES_POS):
            rows.append(row)

    # Tokenization sanity (first 2 rows): MARKER_ID present exactly once.
    for row in rows[:2]:
        full_messages = list(row["prompt"]) + list(row["completion"])
        text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        marker_count = ids.count(MARKER_ID)
        if marker_count != 1:
            raise AssertionError(
                f"cond={cond_id}: encoded training row has {marker_count} marker "
                f"positions, expected 1. Tokenizer may have re-segmented the "
                f"' ※' boundary. First 80 tokens: {ids[:80]}"
            )

    TRAIN_ROW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRAIN_ROW_DIR / f"i462_{cond_id}.jsonl"
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("cond=%s wrote %d positive rows -> %s", cond_id, len(rows), out_path)
    return out_path, rows


class EpochAdapterSaveCallback(TrainerCallback):
    """Persist the LoRA adapter at the END of each listed epoch + upload to HF.

    Why a callback: we need 4 snapshots per condition (epochs 1, 2, 3, 5)
    inside a SINGLE training run so the loss curve is identical to a
    one-shot 5-epoch train (no warm-restart, no checkpoint-resume bias).
    HF Trainer's ``save_strategy='epoch'`` would also write snapshots but
    saves the FULL model state (optimizer, scheduler) and doesn't expose
    a hook to control sub-paths or upload per-epoch. Calling
    ``trainer.model.save_pretrained(dir)`` directly from ``on_epoch_end``
    writes ONLY the PEFT adapter (``trainer.model`` is the wrapped
    PeftModel — see sft.py setting ``peft_config=lora_config`` on
    SFTTrainer construction), which is what we want.

    Saves + uploads are best-effort with respect to upload failure (log
    and continue) but FAIL LOUD on save failure — a partial-snapshot
    pattern would silently lose epoch granularity downstream.
    """

    def __init__(
        self,
        cond_id: str,
        save_epochs: list[int],
        hf_repo: str,
        out_base: Path,
    ) -> None:
        # No super().__init__() — TrainerCallback's base is intentionally
        # empty; if any framework version adds state there, fail loudly
        # (preferable to a silent skip).
        self.cond_id = cond_id
        self.save_epochs = set(save_epochs)
        self.hf_repo = hf_repo
        self.out_base = out_base
        self.saved_epochs: list[int] = []

    def on_epoch_end(self, args, state, control, **kwargs):
        # state.epoch is a float (e.g. 4.9999999 or 5.0). round() returns int.
        epoch_int = round(state.epoch)
        if epoch_int not in self.save_epochs:
            return control

        model = kwargs.get("model")
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        if model is None:
            raise RuntimeError(
                f"EpochAdapterSaveCallback({self.cond_id} ep{epoch_int}): "
                "trainer kwargs missing 'model' — refusing to skip save."
            )

        out_dir = self.out_base / f"i462_{self.cond_id}_ep{epoch_int}"
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "EpochAdapterSaveCallback cond=%s ep=%d (state.epoch=%.4f) -> %s",
            self.cond_id,
            epoch_int,
            state.epoch,
            out_dir,
        )

        # PEFT adapter only — model is the PeftModel wrapper.
        model.save_pretrained(str(out_dir))
        if tokenizer is not None:
            try:
                tokenizer.save_pretrained(str(out_dir))
            except Exception as e:
                # Tokenizer is optional for vLLM LoRA loading; warn and continue.
                logger.warning(
                    "tokenizer.save_pretrained failed for %s: %s (continuing)", out_dir, e
                )

        # Validate the snapshot has the required PEFT files BEFORE upload.
        required = ["adapter_config.json", "adapter_model.safetensors"]
        missing = [f for f in required if not (out_dir / f).exists()]
        if missing:
            raise RuntimeError(
                f"EpochAdapterSaveCallback({self.cond_id} ep{epoch_int}): "
                f"missing PEFT files {missing} in {out_dir} after save_pretrained."
            )

        # Upload to HF Hub at adapters/i462_<cond>_ep{N}.
        try:
            from explore_persona_space.orchestrate.hub import upload_model

            path_in_repo = f"adapters/i462_{self.cond_id}_ep{epoch_int}"
            hub_path = upload_model(
                str(out_dir),
                repo_id=self.hf_repo,
                path_in_repo=path_in_repo,
            )
            if hub_path:
                logger.info("Adapter (ep=%d) uploaded to HF Hub: %s", epoch_int, hub_path)
            else:
                logger.warning(
                    "Adapter (ep=%d) upload failed — local copy preserved at %s",
                    epoch_int,
                    out_dir,
                )
        except Exception as e:
            logger.warning(
                "Adapter (ep=%d) upload failed (%s) — local copy preserved at %s",
                epoch_int,
                e,
                out_dir,
            )

        self.saved_epochs.append(epoch_int)
        return control


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--conds",
        nargs="+",
        required=True,
        help=(
            "One or more condition cids (e.g. A1 or 'A1 A2 B1'). For sweep, list one cid per call."
        ),
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=max(EPOCH_SAVE_POINTS),
        help=(
            "Total training epochs. Default = max(EPOCH_SAVE_POINTS) = 5 so the "
            "epoch-5 snapshot is the FINAL state of a 5-epoch train (matches #460)."
        ),
    )
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help=(
            "PHYSICAL GPU index (Hydra +gpu_id pattern). sft.py sets "
            "os.environ['CUDA_VISIBLE_DEVICES']=str(gpu_id) then loads with "
            "device_map={'':0}, so pass the physical index here (NOT 0 + env "
            "CVD — sft.py clobbers env CVD). Per CLAUDE.md cvd-hydra-override "
            "(#376); the dispatcher passes one physical index per parallel cell."
        ),
    )
    ap.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate (inherited from #406/#460)."
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG + TrainLoraConfig seed.")
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    # MooseFS quota safety per CLAUDE.md gotcha.
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    # Marker assert (mirrors i460).
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]")

    # Validate cond ids.
    unknown = [c for c in args.conds if c not in CONDITIONS_BY_ID]
    if unknown:
        raise ValueError(
            f"--conds {unknown} not in active set {list(CONDITIONS_BY_ID)}. "
            "C2..C5 are dropped per #406 scope change; only A1..A5, B1..B5, C1, D1..D5 are valid."
        )

    if args.epochs < max(EPOCH_SAVE_POINTS):
        raise ValueError(
            f"--epochs={args.epochs} but EPOCH_SAVE_POINTS={EPOCH_SAVE_POINTS} requires "
            f">= {max(EPOCH_SAVE_POINTS)} total epochs."
        )

    q_train_answers = load_q_train_answers()
    class_d_rewrites = load_class_d_rewrites()
    R_train = _load_R("train")

    ADAPTER_OUT_BASE.mkdir(parents=True, exist_ok=True)

    for cond_id in args.conds:
        train_path, _rows = _build_training_rows(
            cond_id, q_train_answers, class_d_rewrites, R_train, tokenizer
        )

        # train_lora needs an output_dir; this is the dir HF Trainer would
        # save_strategy='epoch' write into. With save_strategy='no' and our
        # callback, NO files land here — but train_lora's end-of-train
        # ``trainer.save_model(output_dir)`` still writes the FINAL state.
        # That final state IS the epoch-5 snapshot (already captured by the
        # callback), so we discard this dir post-train to avoid confusion
        # downstream (the canonical epoch-5 adapter is adapters/i462_<cond>_ep5).
        scratch_out_dir = f"adapters/i462_{cond_id}_scratch"
        logger.info(
            "Training cond=%s lr=%s epochs=%d gpu_id=%d marker_only_loss=True "
            "tail_tokens=0 save_points=%s",
            cond_id,
            args.lr,
            args.epochs,
            args.gpu_id,
            EPOCH_SAVE_POINTS,
        )

        cfg = TrainLoraConfig(
            gpu_id=args.gpu_id,
            epochs=args.epochs,
            lr=args.lr,
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.0,
            batch_size=4,
            grad_accum=4,
            max_length=2048,
            seed=args.seed,
            run_name=f"i462_{cond_id}",
            report_to="wandb",
            save_strategy="no",  # callback handles all saves
            marker_only_loss=True,
            marker_text=MARKER_TEXT,
            marker_tail_tokens=0,
            hf_upload=False,  # callback uploads per-epoch; suppress the end-of-train auto-upload
            hf_repo=HF_MODEL_REPO,
            hf_path_in_repo=f"adapters/i462_{cond_id}_scratch",
        )

        callback = EpochAdapterSaveCallback(
            cond_id=cond_id,
            save_epochs=EPOCH_SAVE_POINTS,
            hf_repo=HF_MODEL_REPO,
            out_base=ADAPTER_OUT_BASE,
        )

        out_path, train_loss = train_lora(
            BASE_MODEL,
            train_path,
            scratch_out_dir,
            cfg=cfg,
            callbacks=[callback],
        )
        logger.info(
            "TRAIN DONE cond=%s loss=%.4f saved_epochs=%s (scratch dir at %s)",
            cond_id,
            train_loss,
            callback.saved_epochs,
            out_path,
        )

        # FAIL LOUD if any expected epoch snapshot didn't land — silent
        # missing snapshots would cascade into Phase 4 reading the wrong
        # epoch as "ep=5" (or no-such-cell).
        missing = sorted(set(EPOCH_SAVE_POINTS) - set(callback.saved_epochs))
        if missing:
            raise RuntimeError(
                f"cond={cond_id}: callback never fired for epochs {missing}. "
                f"saved_epochs={callback.saved_epochs}. Either state.epoch did "
                f"not reach those integers (training crashed early?) or the "
                f"callback list was lost — refusing to silently produce a "
                f"partial epoch trajectory."
            )


if __name__ == "__main__":
    main()
