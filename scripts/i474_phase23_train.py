"""Phase 2 (smoke) + Phase 3 (sweep) — train one LoRA per (arm, condition) with
marker-at-end + marker-only loss, with optional broad contrastive negatives.

Issue #474 plan v3 §4.3. Forked from ``scripts/i460_phase23_train.py`` with
three changes:

  B.1  ``--arm pos|loc`` flag and ``_build_negative_rows()`` for A_loc.
       A_loc adds 300 negative rows per source condition (15 bystanders x
       20 sampled Q_train each). Each negative row is
       ``T_j(q) + R_j`` (no marker), tagged ``_neg_source_i`` /
       ``_neg_bystander_j`` for the M5 callback. Under
       ``MarkerOnlyDataCollator(tail_tokens=0,
       suppress_at_post_response_slot=True, im_end_token_id=151645)`` the
       no-marker negative rows train the FIRST ``<|im_end|>`` in the
       completion region (the post-response slot, ``neg_ids[-2]``) — the
       SAME label slot the marker occupies on positives at ``pos_ids[-3]``,
       sharing the same ``...Answer.`` conditioning context, and the slot
       the DV reads. Under softmax competition this pushes ``log P(※)``
       DOWN at the measured slot.

       A_pos has NO negative rows (reproduces #460 byte-identically).
       lora_dropout=0.0 is INHERITED from ``scripts/i460_phase23_train.py``
       (overrides ``TrainLoraConfig`` default 0.05 — see plan v3 §11).

  B.2  ``save_strategy="epoch"`` with adapters saved at epochs 1/2/3/5,
       uploaded to HF under ``adapters/i474_{arm}_{cid}_ep{N}``. Inline
       merged-checkpoint upload is fenced
       (``EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1``) per upload-policy.

  B.3  ``NegRowSuppressionDifficultyCallback`` — M5 identifiability hook.
       At each checkpoint epoch end, computes mean negative-row training
       loss per (source_i, bystander_j) on the trained adapter and writes
       ``eval_results/issue_474/train_diag/suppression_difficulty_{arm}_{cid}_ep{N}.json``.

The vLLM-after-HF Trainer subprocess-isolation contract is inherited
from #460: the smoke-check is invoked as a SEPARATE process by the bash
dispatcher AFTER this script exits (so the OS reaps the HF Trainer's
GPU pin before vLLM tries to init).

CLI (smoke == sweep with --conds A1, plan v3 §4.10 unification):
    # A_pos smoke (reproduces #460 + saves adapters at epochs 1/2/3/5):
    uv run python scripts/i474_phase23_train.py --arm pos --conds A1

    # A_loc smoke (positives + negatives + suppression callback):
    uv run python scripts/i474_phase23_train.py --arm loc --conds A1

    # Sweep cell:
    uv run python scripts/i474_phase23_train.py --arm loc --conds B3 --gpu-id 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, TrainerCallback

from explore_persona_space.experiments.i406_conditions import (
    CONDITIONS,
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

logger = logging.getLogger("i474.phase23")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_R_PATH_PREFIX = "issue460_marker_at_end/on_policy_R"  # SHARED with #460
LOCAL_DATA_DIR = Path("data/issue_460")  # SHARED — same frozen R
TRAIN_ROW_DIR = Path("data/issue_474/train_rows")
M5_OUT_DIR = Path("eval_results/issue_474/train_diag")

# Positive-row count inherited from #460 round-3: 30 Q_train x 10 dupes = 300.
N_DUPES_POS = 10

# Negative-row composition (plan v3 §11):
#   total negatives = 300 (matches positives 1:1, Source: #406 + contrastive-negatives.md)
#   15 bystanders x 20 sampled Q_train each (per-bystander even split, NEW for #474)
N_NEG_PER_BYSTANDER = 20
N_BYSTANDERS_EXPECTED = 15  # 16 conditions - 1 source

# Plan v3 §4.3 B.2: save adapters at these epochs (saved by Trainer at
# step boundaries; epochs are validated by the callback).
DEFAULT_CHECKPOINT_EPOCHS = (1, 2, 3, 5)


def _load_R(split: str) -> dict[str, dict[str, dict]]:
    """Load the frozen R artifact for split in {'train', 'test'}.

    SHARED with #460 — same artifact, no regeneration. Pulls from HF data
    repo if the local file is missing.
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
            f"R_{split}.json schema_version={payload.get('schema_version')!r}, expected 'i460_v1'."
        )
    return payload["completions"]


def _build_messages_for_cond(cond, q: str, class_d_rewrites) -> list[dict]:
    """Construct the chat-message list for one (cond, question).

    Mirrors ``_build_training_rows`` in ``i460_phase23_train.py``. Used by
    BOTH the positive-row builder (R_i, marker appended) and the
    negative-row builder (R_j, NO marker — appended only by the
    completion-text construction in each builder respectively).
    """
    if cond.cls == "A":
        return [
            {"role": "system", "content": cond.system_prompt},
            {"role": "user", "content": q},
        ]
    if cond.cls == "B":
        return [{"role": "user", "content": cond.wrap_template.format(q=q)}]
    if cond.cls == "C" and cond.chat_template:
        return [{"role": "user", "content": q}]
    if cond.cls == "D":
        rewrite = class_d_rewrites[q][cond.register]
        return [{"role": "user", "content": rewrite}]
    raise ValueError(
        f"Unsupported cond {cond.cid} cls={cond.cls} chat_template={cond.chat_template}"
    )


def _build_positive_rows(
    cond_id: str,
    q_train_answers: dict[str, str],
    class_d_rewrites: dict[str, dict[str, str]],
    R_train: dict[str, dict[str, dict]],
    tokenizer,
) -> list[dict]:
    """Build the 30 x N_DUPES_POS positive rows for one condition.

    Identical to ``i460_phase23_train.py:_build_training_rows`` except the
    rows are returned in memory (caller writes the JSONL after combining
    with A_loc negatives).
    """
    cond = CONDITIONS_BY_ID[cond_id]
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
        messages = _build_messages_for_cond(cond, q, class_d_rewrites)
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
                f"cond={cond_id}: encoded POSITIVE row has {marker_count} marker "
                f"positions, expected 1. First 80 tokens: {ids[:80]}"
            )
    return rows


def _build_negative_rows(
    cond_id: str,
    q_train_answers: dict[str, str],
    class_d_rewrites: dict[str, dict[str, str]],
    R_train: dict[str, dict[str, dict]],
    tokenizer,
    rng: np.random.Generator,
) -> list[dict]:
    """Broad contrastive negatives for ``cond_id``.

    For each bystander T_j (≠ ``cond_id``), sample
    ``N_NEG_PER_BYSTANDER=20`` Q_train questions and build a row whose
    completion is ``R_j(q)`` with NO marker. Under
    ``MarkerOnlyDataCollator(tail_tokens=0, suppress_at_post_response_slot=True,
    im_end_token_id=151645)``, no-marker rows get loss ONLY at the FIRST
    ``<|im_end|>`` in the completion region (``neg_ids[-2]``) — the SAME
    label slot the marker occupies on positives at ``pos_ids[-3]``, sharing
    the ``...Answer.`` conditioning context, the SAME slot the DV reads.

    Rows are tagged ``_neg_source_i`` and ``_neg_bystander_j`` for the M5
    callback to group losses by (i, j). The tags do NOT affect the
    training data shape (HF Trainer drops unknown columns); the callback
    sees them on the raw row dicts loaded from JSONL.
    """
    questions = sorted(q_train_answers.keys())
    bystander_ids = [c.cid for c in CONDITIONS if c.cid != cond_id]
    if len(bystander_ids) != N_BYSTANDERS_EXPECTED:
        raise AssertionError(
            f"Expected {N_BYSTANDERS_EXPECTED} bystanders, got {len(bystander_ids)}"
        )

    rows: list[dict] = []
    for cj_id in bystander_ids:
        cj = CONDITIONS_BY_ID[cj_id]
        if cj_id not in R_train:
            raise AssertionError(f"R_train missing bystander {cj_id!r}.")
        sampled = rng.choice(questions, size=N_NEG_PER_BYSTANDER, replace=False)
        for q in sampled:
            R_j = R_train[cj_id][str(q)]["response_text"]
            messages = _build_messages_for_cond(cj, str(q), class_d_rewrites)
            row = {
                "prompt": messages,
                "completion": [{"role": "assistant", "content": R_j}],
                "_neg_source_i": cond_id,
                "_neg_bystander_j": cj_id,
            }
            rows.append(row)

    expected = N_BYSTANDERS_EXPECTED * N_NEG_PER_BYSTANDER
    if len(rows) != expected:
        raise AssertionError(f"expected {expected} negative rows, got {len(rows)}")

    # Tokenization sanity (first 2 negative rows): MARKER_ID absent AND
    # <|im_end|> present in the completion region of the full tokenized text.
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        raise AssertionError("tokenizer cannot resolve <|im_end|>")
    for row in rows[:2]:
        full = list(row["prompt"]) + list(row["completion"])
        text = tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids.count(MARKER_ID) != 0:
            raise AssertionError(
                f"NEGATIVE row contains MARKER_ID; cond_i={cond_id} cj={row['_neg_bystander_j']}"
            )
        if im_end_id not in ids:
            raise AssertionError(
                f"NEGATIVE row has no <|im_end|> ({im_end_id}); "
                f"cond_i={cond_id} cj={row['_neg_bystander_j']} tail ids: {ids[-10:]}"
            )
    return rows


def _write_rows_jsonl(rows: list[dict], out_path: Path) -> None:
    """Write rows to JSONL, stripping fields HF Trainer doesn't accept."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_post_response_slot(
    tokenizer, prompt_messages: list[dict], full_ids: list[int], im_end_id: int
) -> int:
    """Resolve the POST-RESPONSE ``<|im_end|>`` slot for a chat-templated row.

    Plan v3 §4.3 Edit B.3 M5 fix (round 2): Qwen-2.5's chat template emits
    ``<|im_end|>`` at the end of EVERY message — system, user, AND assistant.
    Picking ``full_ids.index(im_end_id)`` returns the SYSTEM-message terminator
    (~position 14 on a 3-message row), so the M5 callback would measure
    ``-log P(<|im_end|> | system prompt)`` instead of the bystander-suppression
    log-prob at the slot the collator + DV both read.

    Robust approach: tokenize ``prompt_messages`` alone WITH
    ``add_generation_prompt=True`` (this includes the assistant-turn opener
    tokens such as ``<|im_start|>assistant\\n``), take its length ``P``,
    then find the first ``<|im_end|>`` at index ``>= P``. That slot is the
    assistant-turn terminator — the SAME slot the marker occupies on positives
    at ``pos_ids[-3]`` (positives append ``<marker><|im_end|>\\n``) and the
    SAME slot the DV reads at ``len(eval_full_ids) - 1`` after the eval-time
    ``prompt_text + R_text + MARKER_TEXT`` byte-exact encoding.

    Returns the integer index of the post-response ``<|im_end|>`` slot.

    Raises:
        RuntimeError: if no ``<|im_end|>`` is found at index ``>= P``, or if
            the prompt-only encoding is not a strict prefix of the full
            encoding (chat-template drift), or if the resolved slot is NOT
            strictly greater than the first ``<|im_end|>`` in the row (a
            cross-check that the slot is in the completion, not the system /
            user region).
    """
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    P = len(prompt_ids)
    if full_ids[:P] != prompt_ids:
        raise RuntimeError(
            "prompt-only encoding is not a strict prefix of the full row encoding "
            f"(chat-template drift). prompt_ids[:5]={prompt_ids[:5]}, "
            f"full_ids[:5]={full_ids[:5]}, P={P}"
        )
    # First <|im_end|> at index >= P is the assistant-turn terminator
    # (the post-response slot).
    slot = next((i for i in range(P, len(full_ids)) if full_ids[i] == im_end_id), None)
    if slot is None:
        raise RuntimeError(
            f"no <|im_end|> (id={im_end_id}) found at index >= P={P} in row of "
            f"length {len(full_ids)}; tail ids: {full_ids[-10:]}"
        )
    # Cross-check: slot must be strictly past the first transcript <|im_end|>.
    # Picking the system-message terminator was the v1 M5 bug; this assertion
    # is the second line of defence.
    first_im_end = next((i for i, t in enumerate(full_ids) if t == im_end_id), None)
    if first_im_end is None or slot <= first_im_end:
        raise RuntimeError(
            "post-response slot resolution returned a slot at or before the first "
            f"<|im_end|> in the row: slot={slot}, first_im_end={first_im_end}, P={P}, "
            f"len(full_ids)={len(full_ids)} — this is the v1 M5 bug class."
        )
    return slot


class NegRowSuppressionDifficultyCallback(TrainerCallback):
    """Plan v3 §4.3 Edit B.3 — M5 identifiability hook.

    At each checkpoint epoch end, computes mean negative-row training loss
    per (source_i, bystander_j) on the trained adapter at that checkpoint
    and writes to JSON. Zero-extra-compute beyond a forward pass on the
    300 negative rows.

    Loss surface: at the POST-RESPONSE ``<|im_end|>`` slot (the assistant-turn
    terminator) — the SAME slot the collator masks loss to on these negative
    rows and the SAME slot the DV reads. The slot is resolved via
    ``_resolve_post_response_slot``, NOT ``list.index`` — Qwen-2.5 emits
    ``<|im_end|>`` after every message turn, so ``ids.index(im_end_id)``
    returns the SYSTEM-message terminator (the v1 M5 bug).

    The loss is ``-log P(<|im_end|> | preceding tokens)`` at that slot.
    """

    def __init__(
        self,
        tokenizer,
        neg_rows: list[dict],
        im_end_id: int,
        arm: str,
        cid: str,
        out_dir: Path,
    ):
        self.tokenizer = tokenizer
        self.neg_rows = neg_rows
        self.im_end_id = im_end_id
        self.arm = arm
        self.cid = cid
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Pre-tokenize each negative row once + pre-resolve the post-response
        # slot. Store (ids, slot, source_i, bystander_j) per row.
        self._cached_rows: list[tuple[list[int], int, str, str]] = []
        for row in neg_rows:
            messages = list(row["prompt"]) + list(row["completion"])
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            ids = tokenizer.encode(text, add_special_tokens=False)
            slot = _resolve_post_response_slot(tokenizer, list(row["prompt"]), ids, im_end_id)
            # Sanity: assert the token at the resolved slot IS im_end_id.
            if ids[slot] != im_end_id:
                raise RuntimeError(
                    f"M5 slot picker returned slot={slot} but ids[slot]={ids[slot]} "
                    f"!= im_end_id={im_end_id}; cid={cid} bystander={row['_neg_bystander_j']}"
                )
            self._cached_rows.append((ids, slot, row["_neg_source_i"], row["_neg_bystander_j"]))

    def on_save(self, args, state, control, **kwargs):
        if self.arm != "loc":
            return
        model = kwargs.get("model")
        if model is None:
            logger.warning("M5 callback: model kwarg missing on on_save; skipping checkpoint.")
            return

        per_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
        model.eval()
        with torch.no_grad():
            for ids, slot, source_i, bystander_j in self._cached_rows:
                input_ids = torch.tensor([ids[: slot + 1]], device=model.device, dtype=torch.long)
                out = model(input_ids=input_ids)
                logp = F.log_softmax(out.logits[0, slot - 1], dim=-1)
                loss_at_slot = -float(logp[self.im_end_id].item())
                per_pair[(source_i, bystander_j)].append(loss_at_slot)
        model.train()

        agg = {f"{i}__{j}": float(np.mean(losses)) for (i, j), losses in per_pair.items()}
        epoch = state.epoch if state.epoch is not None else float("nan")
        out_path = self.out_dir / f"suppression_difficulty_{self.arm}_{self.cid}_ep{epoch:g}.json"
        out_path.write_text(
            json.dumps(
                {
                    "arm": self.arm,
                    "source_i": self.cid,
                    "epoch": epoch,
                    "global_step": state.global_step,
                    "per_bystander_mean_neg_loss": agg,
                },
                indent=2,
            )
        )
        logger.info(
            "M5 suppression_difficulty arm=%s cid=%s ep=%g step=%d pairs=%d -> %s",
            self.arm,
            self.cid,
            epoch,
            state.global_step,
            len(agg),
            out_path,
        )
        # Best-effort WandB log.
        if "wandb" in args.report_to:
            try:
                import wandb

                wandb.log(
                    {f"neg_loss/{self.cid}__{k.split('__')[1]}": v for k, v in agg.items()},
                    step=int(state.global_step),
                )
            except Exception as e:
                logger.warning("M5 wandb.log failed (non-fatal): %s", e)


class PerEpochAdapterHFUploadCallback(TrainerCallback):
    """Plan v3 §4.3 Edit B.2 — per-epoch HF adapter upload (round-2 fix).

    Round-1 bug: the train script wrote ``save_strategy="epoch"`` (so
    ``checkpoint-<step>/`` dirs landed on local disk per epoch), but the
    only HF push was the SINGLE end-of-training upload at
    ``adapters/i474_{arm}_{cid}`` (no ``_ep{N}``). Phase 4 eval +
    Phase 2 smoke read ``adapters/i474_{arm}_{cid}_ep{N}`` for
    N in {1,2,3,5} → the across-epoch sweep (the WHOLE point of the
    re-run — the epoch-1 saturation knee) was dead on arrival.

    This callback fires at every ``on_save`` (i.e. every epoch under
    ``save_strategy="epoch"``). When ``state.epoch`` is in
    ``CHECKPOINT_EPOCHS_TO_UPLOAD`` it:

      1. Reads the freshly-written ``checkpoint-<state.global_step>/``
         directory (the parent ``Trainer._save_checkpoint`` already wrote
         ``adapter_model.safetensors`` + ``adapter_config.json`` there).
      2. Copies the tokenizer files (saved once in ``output_dir`` by
         SFTTrainer at init via ``processing_class=tokenizer``) INTO the
         checkpoint dir so the uploaded bundle is self-contained for
         vLLM LoRA load.
      3. Uploads the checkpoint dir to
         ``adapters/i474_{arm}_{cid}_ep{N}`` via the shared
         ``upload_model`` helper. Sets
         ``EPM_PERSIST_ADAPTER_HF_REPO`` /
         ``EPM_PERSIST_ADAPTER_SUBFOLDER`` per
         ``.claude/rules/upload-policy.md`` so the contract is explicit.
      4. **Fail-loud on upload failure (raises)** — the launcher's
         ``set -e`` aborts the cell BEFORE any later local deletion, per
         upload-policy.md.

    Never uploads the merged 15GB dir (only the adapter). The
    ``EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1`` fence stays on in main() so
    ``train_lora``'s end-of-training upload is the ONLY non-callback
    push, and it goes to the bare ``adapters/i474_{arm}_{cid}`` path
    (the final-epoch convenience copy).
    """

    # Plan v3 §4.3 B.2: epochs to upload per condition.
    CHECKPOINT_EPOCHS_TO_UPLOAD: tuple[int, ...] = (1, 2, 3, 5)

    def __init__(
        self,
        arm: str,
        cid: str,
        output_dir: str,
        hf_repo: str = HF_MODEL_REPO,
    ):
        self.arm = arm
        self.cid = cid
        self.output_dir = Path(output_dir)
        self.hf_repo = hf_repo
        self._uploaded_epochs: set[int] = set()

    @staticmethod
    def _resolve_target_epoch(state_epoch: float | None) -> int | None:
        """Map fractional ``state.epoch`` to an integer target epoch.

        HF Trainer reports ``state.epoch`` as a float just past the epoch
        boundary (e.g. 1.0, 2.0, 3.0, 5.0). Round to the nearest int and
        return iff it is in CHECKPOINT_EPOCHS_TO_UPLOAD.
        """
        if state_epoch is None:
            return None
        candidate = round(state_epoch)
        # Float epsilon guard: state.epoch can be 0.9999 just before save.
        if abs(state_epoch - candidate) > 0.05:
            return None
        if candidate in PerEpochAdapterHFUploadCallback.CHECKPOINT_EPOCHS_TO_UPLOAD:
            return candidate
        return None

    # Round-3 fix: upload bundle is ONLY the adapter + tokenizer files
    # the eval/smoke download paths actually need. EXCLUDE optimizer.pt,
    # rng_state.pth, scheduler.pt, trainer_state.json, training_args.bin
    # (HF Trainer's full-state files; for 7B LoRA at r=32 the optimizer.pt
    # alone is ~hundreds of MB and is useless for inference / vLLM LoRA load).
    # KEEP IN SYNC with i474_phase4_eval.py:_download_adapters and
    # i474_phase2_smoke_check.py:_resolve_adapter_path required-file lists.
    UPLOAD_ALLOWLIST: tuple[str, ...] = (
        # Required by eval + smoke downloads:
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        # Optional but cheap + sometimes needed by tokenizer init across versions:
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
        "chat_template.jinja",
        "README.md",
    )
    UPLOAD_EXCLUDED: tuple[str, ...] = (
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    )

    def _checkpoint_dir(self, global_step: int) -> Path:
        return self.output_dir / f"checkpoint-{global_step}"

    def _stage_clean_upload_bundle(self, checkpoint_dir: Path, target_ep: int) -> Path:
        """Assemble a clean upload-only directory with the allowlisted files.

        Builds ``output_dir/_upload_ep{N}/`` from:
          - ``checkpoint-<step>/`` (adapter files)
          - ``output_dir/`` (tokenizer files written by SFTTrainer at init)

        Filters to ``UPLOAD_ALLOWLIST`` so ``optimizer.pt`` / ``rng_state.pth`` /
        ``scheduler.pt`` / ``trainer_state.json`` / ``training_args.bin`` from
        the checkpoint dir are NEVER copied. ``shutil.copy2`` preserves the
        adapter mtime so the verification round-trip lines up.

        Returns the upload-only directory path. The caller is responsible
        for cleaning it up post-upload (we leave it on disk so a failed
        upload can be retried by hand).
        """
        import shutil

        upload_dir = self.output_dir / f"_upload_ep{target_ep}"
        # Idempotent: a retry on the same epoch wipes the stale stage.
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=False)

        copied: list[str] = []
        excluded_seen: list[str] = []
        # 1. Pull adapter files from the checkpoint dir.
        for fname in self.UPLOAD_ALLOWLIST:
            src = checkpoint_dir / fname
            if src.exists():
                shutil.copy2(src, upload_dir / fname)
                copied.append(f"ckpt:{fname}")
        for fname in self.UPLOAD_EXCLUDED:
            if (checkpoint_dir / fname).exists():
                excluded_seen.append(f"ckpt:{fname}")
        # 2. Pull tokenizer files from output_dir (SFTTrainer writes them
        #    once at init via processing_class=tokenizer); only fill in
        #    files NOT already pulled from the checkpoint dir.
        for fname in self.UPLOAD_ALLOWLIST:
            src = self.output_dir / fname
            dst = upload_dir / fname
            if src.exists() and not dst.exists() and src.is_file():
                shutil.copy2(src, dst)
                copied.append(f"out:{fname}")

        if "adapter_model.safetensors" not in [c.split(":", 1)[1] for c in copied]:
            raise RuntimeError(
                f"PerEpochAdapterHFUpload: clean upload bundle missing "
                f"adapter_model.safetensors after staging from {checkpoint_dir} "
                f"and {self.output_dir}. Copied: {copied}"
            )
        if "adapter_config.json" not in [c.split(":", 1)[1] for c in copied]:
            raise RuntimeError(
                f"PerEpochAdapterHFUpload: clean upload bundle missing "
                f"adapter_config.json after staging from {checkpoint_dir}. "
                f"Copied: {copied}"
            )

        logger.info(
            "PerEpochAdapterHFUpload: staged upload bundle ep=%d at %s "
            "(%d files copied; %d excluded files seen in checkpoint dir): "
            "copied=%s excluded_seen=%s",
            target_ep,
            upload_dir,
            len(copied),
            len(excluded_seen),
            copied,
            excluded_seen,
        )
        return upload_dir

    def on_save(self, args, state, control, **kwargs):
        target_ep = self._resolve_target_epoch(state.epoch)
        if target_ep is None:
            logger.debug(
                "PerEpochAdapterHFUpload: state.epoch=%s not in %s; skipping.",
                state.epoch,
                self.CHECKPOINT_EPOCHS_TO_UPLOAD,
            )
            return
        if target_ep in self._uploaded_epochs:
            logger.debug("PerEpochAdapterHFUpload: ep%d already uploaded; skipping.", target_ep)
            return

        ckpt_dir = self._checkpoint_dir(state.global_step)
        adapter_file = ckpt_dir / "adapter_model.safetensors"
        config_file = ckpt_dir / "adapter_config.json"
        if not adapter_file.exists() or not config_file.exists():
            raise RuntimeError(
                f"PerEpochAdapterHFUpload: expected adapter files missing under "
                f"{ckpt_dir} (adapter_model.safetensors / adapter_config.json). "
                "Trainer.save_strategy='epoch' did not produce them — refusing to "
                "silently skip the per-epoch upload (Phase 4 eval would fail-loud "
                "on _ep{N} download). Check that PEFT is wrapping the model."
            )
        upload_dir = self._stage_clean_upload_bundle(ckpt_dir, target_ep)

        # The path-in-repo contract Phase 4 + smoke read from. KEEP IN
        # SYNC with i474_phase4_eval.py::_download_adapters and
        # i474_phase2_smoke_check.py::_resolve_adapter_path.
        path_in_repo = f"adapters/i474_{self.arm}_{self.cid}_ep{target_ep}"

        # Explicit env contract per upload-policy.md (so other surfaces
        # that read these env vars know where the adapter persisted).
        os.environ["EPM_PERSIST_ADAPTER_HF_REPO"] = self.hf_repo
        os.environ["EPM_PERSIST_ADAPTER_SUBFOLDER"] = path_in_repo

        from explore_persona_space.orchestrate.hub import upload_model

        logger.info(
            "PerEpochAdapterHFUpload: arm=%s cid=%s ep=%d step=%d uploading %s -> %s/%s",
            self.arm,
            self.cid,
            target_ep,
            state.global_step,
            upload_dir,
            self.hf_repo,
            path_in_repo,
        )
        hub_path = upload_model(
            str(upload_dir),
            repo_id=self.hf_repo,
            path_in_repo=path_in_repo,
        )
        if not hub_path:
            raise RuntimeError(
                "PerEpochAdapterHFUpload: upload_model returned empty string "
                f"(verification failed) for arm={self.arm} cid={self.cid} ep={target_ep} "
                f"-> {self.hf_repo}/{path_in_repo}. Per upload-policy.md fail-loud "
                "contract, refusing to continue training — the local checkpoint will "
                "be reaped by the next save_total_limit cycle and the sweep will lose "
                "this epoch's adapter."
            )
        self._uploaded_epochs.add(target_ep)
        logger.info(
            "PerEpochAdapterHFUpload: arm=%s cid=%s ep=%d uploaded to %s (verified).",
            self.arm,
            self.cid,
            target_ep,
            hub_path,
        )

        # ─────────────────────────────────────────────────────────────────
        # Round-5 FIX A — delete local checkpoint dir + staged upload bundle
        # AFTER verified HF upload. The round-3 fix made the upload bundle
        # adapter-only but left the source ``checkpoint-{step}/`` dirs on
        # disk (each ~1.8GB with optimizer.pt/rng_state/scheduler). At
        # ~84 conditions x 5 epochs that's 756 GB locally — far past the
        # MooseFS per-pod ~130 GB quota → EDQUOT → silent SIGKILL mid-sweep.
        #
        # Order matters per upload-policy.md fail-loud contract:
        #   1. upload + verify (above) — if verification failed we raised
        #      already and never reach this block.
        #   2. delete staged ``_upload_ep{N}/`` bundle (the disposable copy).
        #   3. delete source ``checkpoint-{step}/`` dir (the heavyweight one).
        #
        # We do NOT delete unverified checkpoints — the raise above is the
        # safety. We also do NOT touch the parent ``output_dir`` (it holds
        # the tokenizer + the FINAL-epoch adapter that ``train_lora``'s
        # end-of-training hub_upload also pushes; reaping it would break
        # the byte-identity contract that final-adapter upload depends on).
        # ─────────────────────────────────────────────────────────────────
        import shutil

        for path, label in ((upload_dir, "upload bundle"), (ckpt_dir, "checkpoint dir")):
            try:
                if path.exists():
                    shutil.rmtree(path)
                    logger.info(
                        "PerEpochAdapterHFUpload: reaped local %s %s "
                        "(arm=%s cid=%s ep=%d) — HF copy at %s is the source of truth now.",
                        label,
                        path,
                        self.arm,
                        self.cid,
                        target_ep,
                        hub_path,
                    )
            except OSError as e:
                # Fail-loud: if we can't reap, the disk fills and the sweep
                # dies the same way. Surface the error so the operator can
                # intervene (e.g. permissions, NFS hiccup) BEFORE the next
                # epoch's checkpoint compounds the problem.
                raise RuntimeError(
                    f"PerEpochAdapterHFUpload: FAILED to reap local {label} {path} "
                    f"after verified HF upload (arm={self.arm} cid={self.cid} "
                    f"ep={target_ep}): {e}. Disk will fill — refusing to continue. "
                    "Operator must investigate (permissions / mount / quota)."
                ) from e


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--arm",
        required=True,
        choices=["pos", "loc"],
        help="pos: positives-only (reproduces #460). loc: positives + 300 contrastive negatives.",
    )
    ap.add_argument(
        "--conds",
        nargs="+",
        required=True,
        help="One or more condition cids (e.g. A1 or 'A1 A2 B1').",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Inherited from #460 round-3 (5).",
    )
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help=(
            "PHYSICAL GPU index (Hydra +gpu_id pattern). sft.py sets "
            "os.environ['CUDA_VISIBLE_DEVICES']=str(gpu_id). Per CLAUDE.md cvd-hydra-override."
        ),
    )
    ap.add_argument("--lr", type=float, default=1e-5, help="Inherited from #460.")
    ap.add_argument("--seed", type=int, default=42, help="Inherited from #460.")
    ap.add_argument(
        "--save-strategy",
        default="epoch",
        choices=["epoch", "no"],
        help=(
            "Default 'epoch' saves adapters at each epoch boundary (plan §4.3 B.2 "
            "saves epochs 1/2/3/5 per condition)."
        ),
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    # MooseFS quota safety per CLAUDE.md gotcha + upload-policy.md adapter-persist.
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    # Marker assert.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id != 151645:
        raise AssertionError(f"<|im_end|> id drift: got {im_end_id}, expected 151645")

    unknown = [c for c in args.conds if c not in CONDITIONS_BY_ID]
    if unknown:
        raise ValueError(
            f"--conds {unknown} not in active set {list(CONDITIONS_BY_ID)}. "
            "C2..C5 are dropped per #406 scope change."
        )

    q_train_answers = load_q_train_answers()
    class_d_rewrites = load_class_d_rewrites()
    R_train = _load_R("train")

    for cond_id in args.conds:
        # Stable per-condition seed offset. Python's built-in hash() is
        # process-randomized unless PYTHONHASHSEED is fixed, so the v1
        # `args.seed + hash(cond_id) % 10_000` did NOT give reproducible
        # A_loc negative rows across launches. Use sha256 of the cond_id
        # bytes for a process-stable offset.
        cond_offset = int(hashlib.sha256(cond_id.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(args.seed + cond_offset % 10_000)
        pos_rows = _build_positive_rows(
            cond_id, q_train_answers, class_d_rewrites, R_train, tokenizer
        )
        if len(pos_rows) != 30 * N_DUPES_POS:
            raise AssertionError(f"expected {30 * N_DUPES_POS} pos rows, got {len(pos_rows)}")

        if args.arm == "loc":
            neg_rows = _build_negative_rows(
                cond_id, q_train_answers, class_d_rewrites, R_train, tokenizer, rng
            )
            expected_neg = N_BYSTANDERS_EXPECTED * N_NEG_PER_BYSTANDER
            if len(neg_rows) != expected_neg:
                raise AssertionError(f"expected {expected_neg} neg rows, got {len(neg_rows)}")
            all_rows = pos_rows + neg_rows
            ratio = len(pos_rows) / max(1, len(neg_rows))
            logger.info(
                "cond=%s arm=loc rows: %d pos + %d neg = %d (ratio pos:neg = %.2f:1)",
                cond_id,
                len(pos_rows),
                len(neg_rows),
                len(all_rows),
                ratio,
            )
        else:
            neg_rows = []
            all_rows = pos_rows
            logger.info(
                "cond=%s arm=pos rows: %d pos (no negatives — reproduces #460)",
                cond_id,
                len(pos_rows),
            )

        train_path = TRAIN_ROW_DIR / f"i474_{args.arm}_{cond_id}.jsonl"
        _write_rows_jsonl(all_rows, train_path)

        out_dir = f"adapters/i474_{args.arm}_{cond_id}"
        logger.info(
            "Training cond=%s arm=%s lr=%s epochs=%d gpu_id=%d save_strategy=%s "
            "marker_only_loss=True tail_tokens=0 "
            "suppress_at_post_response_slot=%s im_end_id=%d",
            cond_id,
            args.arm,
            args.lr,
            args.epochs,
            args.gpu_id,
            args.save_strategy,
            args.arm == "loc",
            im_end_id,
        )

        callbacks: list[TrainerCallback] = []
        # BOTH arms get the per-epoch HF adapter upload — Phase 4 + smoke
        # read adapters/i474_{arm}_{cid}_ep{N} for N in {1,2,3,5}.
        if args.save_strategy == "epoch":
            callbacks.append(
                PerEpochAdapterHFUploadCallback(
                    arm=args.arm,
                    cid=cond_id,
                    output_dir=out_dir,
                )
            )
        if args.arm == "loc":
            callbacks.append(
                NegRowSuppressionDifficultyCallback(
                    tokenizer=tokenizer,
                    neg_rows=neg_rows,
                    im_end_id=im_end_id,
                    arm=args.arm,
                    cid=cond_id,
                    out_dir=M5_OUT_DIR,
                )
            )

        cfg = TrainLoraConfig(
            gpu_id=args.gpu_id,
            epochs=args.epochs,
            lr=args.lr,
            lora_r=32,
            lora_alpha=64,
            # Inherited from scripts/i460_phase23_train.py @ 15c99ae6 — A_pos
            # must reproduce what trained #460 (the i460 script value, NOT
            # the TrainLoraConfig default 0.05). See plan v3 §11.
            lora_dropout=0.0,
            batch_size=4,
            grad_accum=4,
            max_length=2048,
            seed=args.seed,
            run_name=f"i474_{args.arm}_{cond_id}",
            report_to="wandb",
            save_strategy=args.save_strategy,
            # Round-5 FIX A: keep at most 1 local checkpoint at a time. The
            # PerEpochAdapterHFUploadCallback reaps each epoch's checkpoint
            # right after verified HF upload, so this is a belt-and-braces
            # backstop: HF Trainer auto-prunes older checkpoints when a
            # new one lands. Was ``None`` (unlimited) → accumulated 5x at
            # ~1.8GB each per condition → 84 conds x 9GB = ~750GB local,
            # blew the MooseFS per-pod ~130GB quota → silent SIGKILL.
            save_total_limit=1,
            marker_only_loss=True,
            marker_text=MARKER_TEXT,
            marker_tail_tokens=0,
            # NEW — only the A_loc arm enables the suppression branch.
            marker_suppress_at_post_response_slot=(args.arm == "loc"),
            marker_im_end_token_id=im_end_id,
            # #628 legacy pin: #474's loc arm trained suppress-ON negatives
            # WITHOUT the trailing-token keep; keep masks byte-identical.
            marker_negative_keep_trailing=False,
            hf_upload=True,
            hf_repo=HF_MODEL_REPO,
            hf_path_in_repo=f"adapters/i474_{args.arm}_{cond_id}",
        )

        out_path, train_loss = train_lora(
            BASE_MODEL, train_path, out_dir, cfg=cfg, callbacks=callbacks
        )
        logger.info(
            "TRAIN DONE arm=%s cond=%s loss=%.4f -> %s",
            args.arm,
            cond_id,
            train_loss,
            out_path,
        )


if __name__ == "__main__":
    main()
