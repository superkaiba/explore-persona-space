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


class NegRowSuppressionDifficultyCallback(TrainerCallback):
    """Plan v3 §4.3 Edit B.3 — M5 identifiability hook.

    At each checkpoint epoch end, computes mean negative-row training loss
    per (source_i, bystander_j) on the trained adapter at that checkpoint
    and writes to JSON. Zero-extra-compute beyond a forward pass on the
    300 negative rows.

    Loss surface: at the FIRST ``<|im_end|>`` in the completion region of
    each negative row (the same slot the collator masks loss to). The
    loss is ``-log P(<|im_end|> | preceding tokens)``.
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
        # Pre-tokenize each negative row once (full chat-template text);
        # store as a parallel list of (input_ids[int], source_i, bystander_j).
        self._cached_rows: list[tuple[list[int], str, str]] = []
        for row in neg_rows:
            messages = list(row["prompt"]) + list(row["completion"])
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            ids = tokenizer.encode(text, add_special_tokens=False)
            self._cached_rows.append((ids, row["_neg_source_i"], row["_neg_bystander_j"]))

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
            for ids, source_i, bystander_j in self._cached_rows:
                if self.im_end_id not in ids:
                    continue
                slot = ids.index(self.im_end_id)
                if slot == 0:
                    continue
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
        rng = np.random.default_rng(args.seed + hash(cond_id) % 10_000)
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
            save_total_limit=None,
            marker_only_loss=True,
            marker_text=MARKER_TEXT,
            marker_tail_tokens=0,
            # NEW — only the A_loc arm enables the suppression branch.
            marker_suppress_at_post_response_slot=(args.arm == "loc"),
            marker_im_end_token_id=im_end_id,
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
