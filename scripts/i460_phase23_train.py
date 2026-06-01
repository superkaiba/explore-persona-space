"""Phase 2 (smoke) + Phase 3 (sweep) — train one LoRA per condition with
marker-at-end + marker-only loss.

Issue #460 plan v3 §4.2 + §4.4 + §4.5.

Per condition T_i:
  - Read R_i(q) from the frozen ``data/issue_460/R_train.json`` artifact
    (Phase 1 output; train↔eval consistency contract).
  - Build 30 positive rows of shape
      prompt    = build_prompt_for_condition(T_i, q)        # chat-template
      completion = R_i(q) + ' ※'                            # marker at END
    The completion is a single assistant turn; TRL's prompt-completion
    format applies response-only masking, then MarkerOnlyDataCollator
    (tail_tokens=0) further masks every R token to -100 — loss lands ONLY
    on the marker token + EOS.
  - NO negative rows (plan §11 / A15: marker-only loss has zero gradient
    on rows without a marker, so negatives are signal-free).
  - LoRA recipe inherited from #406: r=32, alpha=64, lr=1e-5, bf16, 3 epochs.
  - Smoke = same script with ``--conds A1 --epochs 1`` (architectural
    parity per plan §4.8).

Smoke implant verification (Phase 2 Gate c, ≥80% held-out implant) was
SPLIT OUT 2026-06-01 (round-2 fix) into ``i460_phase2_smoke_check.py``,
a separate subprocess invoked by the bash dispatcher AFTER this script
exits. In-process vLLM-after-HF-Trainer triggered "model already on
multiple devices" / EngineCore ``init_device`` failure — the documented
CLAUDE.md gotcha (task #399). Subprocess isolation = OS reaps the HF
Trainer's GPU pin before vLLM tries to init.

CLI:
    # Smoke (Phase 2) — 1-epoch train only; dispatcher runs smoke-check after:
    uv run python scripts/i460_phase23_train.py --conds A1 --epochs 1

    # Single-condition sweep dispatcher cell:
    uv run python scripts/i460_phase23_train.py --conds A2 --epochs 3 --gpu-id 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from transformers import AutoTokenizer

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

logger = logging.getLogger("i460.phase23")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_R_PATH_PREFIX = "issue460_marker_at_end/on_policy_R"
LOCAL_DATA_DIR = Path("data/issue_460")
TRAIN_ROW_DIR = Path("data/issue_460/train_rows")
# NOTE: smoke-implant constants + adapter-cache + smoke-log-dir live in
# scripts/i460_phase2_smoke_check.py now (round-2 subprocess-isolation fix).


def _load_R(split: str) -> dict[str, dict[str, dict]]:
    """Load the frozen R artifact for split in {'train', 'test'}.

    Pulls from HF data repo if the local file is missing.
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
    """Build the 30 positive rows for one condition and write a JSONL on disk.

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
        rows.append(
            {
                "prompt": messages,
                "completion": [{"role": "assistant", "content": completion_text}],
            }
        )

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
    out_path = TRAIN_ROW_DIR / f"i460_{cond_id}.jsonl"
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("cond=%s wrote %d positive rows -> %s", cond_id, len(rows), out_path)
    return out_path, rows


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
    ap.add_argument("--epochs", type=int, default=3, help="3 for sweep, 1 for smoke.")
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help=(
            "Local GPU id (Hydra +gpu_id pattern). The bash dispatcher sets "
            "env CUDA_VISIBLE_DEVICES=<phys>; this remaps the visible GPU to "
            "local id 0. Per CLAUDE.md cvd-hydra-override (#376)."
        ),
    )
    ap.add_argument("--lr", type=float, default=1e-5, help="Learning rate (inherited from #406).")
    ap.add_argument("--seed", type=int, default=42, help="RNG + TrainLoraConfig seed.")
    # NOTE: --smoke-eval was REMOVED 2026-06-01 (round-2 fix). vLLM-after-HF in
    # the same process triggers "model already on multiple devices" / EngineCore
    # init_device failure (CLAUDE.md gotcha, task #399). The dispatcher now
    # invokes scripts/i460_phase2_smoke_check.py as a SEPARATE process AFTER
    # this script exits, so the OS reaps the HF Trainer's GPU pin before vLLM
    # tries to init.
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    # MooseFS quota safety per CLAUDE.md gotcha.
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    # Marker assert.
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

    q_train_answers = load_q_train_answers()
    class_d_rewrites = load_class_d_rewrites()
    R_train = _load_R("train")

    for cond_id in args.conds:
        train_path, _rows = _build_training_rows(
            cond_id, q_train_answers, class_d_rewrites, R_train, tokenizer
        )

        out_dir = f"adapters/i460_{cond_id}"
        logger.info(
            "Training cond=%s lr=%s epochs=%d gpu_id=%d marker_only_loss=True tail_tokens=0",
            cond_id,
            args.lr,
            args.epochs,
            args.gpu_id,
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
            run_name=f"i460_{cond_id}",
            report_to="wandb",
            save_strategy="no",
            marker_only_loss=True,
            marker_text=MARKER_TEXT,
            marker_tail_tokens=0,
            hf_upload=True,
            hf_repo=HF_MODEL_REPO,
            hf_path_in_repo=f"adapters/i460_{cond_id}",
        )

        out_path, train_loss = train_lora(BASE_MODEL, train_path, out_dir, cfg=cfg)
        logger.info("TRAIN DONE cond=%s loss=%.4f -> %s", cond_id, train_loss, out_path)
        # Smoke implant check is now a separate subprocess invoked by the
        # dispatcher AFTER this script exits — see i460_phase2_smoke_check.py
        # and the FIX-1 note above on vLLM-after-HF in-process conflict.


if __name__ == "__main__":
    main()
