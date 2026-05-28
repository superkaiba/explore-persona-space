"""Phase 2 — train one LoRA per transformation (sharded entry, one cond per call).

Issue #406 plan v9 §4 Phase 2.

Builds the 600-row training mix for `--condition <cid>` and invokes
`train_lora()` from src/explore_persona_space/train/sft.py with the
correct config per condition:

  - Class A / B / D / C1 — prompt-completion JSONL, response-only loss
    masking (default SFTTrainer behavior). 3 epochs, lr=1e-5.
  - Class C2 / C3 / C4 / C5 — raw-text JSONL, full-sequence loss via
    `dataset_text_field="text"` (the v3 MF-2 trainer extension), 1 epoch,
    lr=5e-6. The training_text rows MUST place ` ※` (id 83399) at the
    position immediately after the literal `Answer:` token; the trainer's
    new `_audit_marker_in_loss_mask()` preflight verifies this on rows
    0 and 1 before training starts.

Row construction per T_i:
  - 30 Q_train questions, each duplicated 10x with T_i shape +
    ' ※\\n\\n<claude_answer>' completion = 300 positive rows.
  - 300 negative rows where the same 30 questions are wrapped under a
    randomly-sampled T_k != T_i (10x per question, distributed across
    the 19 other conditions) with the matching <claude_answer> completion
    and NO marker.

Tokenization sanity assert at launch: each of the first 2 training rows'
` ※` (id 83399) lands exactly where expected (after `Answer:` for C2..C5,
or as the first assistant token for chat-template conditions).

CLI:
    uv run python scripts/i406_phase2_train_loras.py --condition A1 --gpu-id 0
    uv run python scripts/i406_phase2_train_loras.py --condition C2 --gpu-id 0 --override-lr 2.5e-6
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from transformers import AutoTokenizer

from explore_persona_space.experiments.i406_conditions import (
    CONDITIONS,
    CONDITIONS_BY_ID,
    MARKER_ID,
    MARKER_TEXT,
    build_prompt_for_condition,
)
from explore_persona_space.train.sft import TrainLoraConfig, train_lora

logger = logging.getLogger("i406.phase2")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_REPO = "superkaiba1/explore-persona-space"
N_DUPES_POS = 10  # 30 questions x 10 dupes = 300 positive rows
N_DUPES_NEG = 10  # 30 questions x 10 dupes = 300 negative rows
TRAIN_DATA_DIR = Path("data/issue_406/train_rows")


def _load_q_train_answers() -> dict[str, str]:
    """Load the 30 Claude-generated Q_train answers from Phase 0."""
    path = Path("data/issue_406/q_train_answers.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Q_train answers not found at {path}. Run scripts/i406_phase0_generate_data.py first."
        )
    with open(path) as f:
        return json.load(f)


def _load_class_d_rewrites() -> dict[str, dict[str, str]]:
    """Load Class D rewrites from Phase 0."""
    path = Path("data/issue_406/class_d/rewrites_v1.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Class D rewrites not found at {path}. Run scripts/i406_phase0_generate_data.py first."
        )
    with open(path) as f:
        return json.load(f)


def _build_training_rows(
    cond_id: str,
    q_train_answers: dict[str, str],
    class_d_rewrites: dict[str, dict[str, str]],
    tokenizer,
    seed: int,
) -> tuple[list[dict], Path]:
    """Build positive + negative training rows for one condition.

    Returns (rows, path) where path is the per-condition JSONL on disk.

    For Class A / B / D / C1: rows are prompt-completion shape.
    For Class C2..C5: rows are raw-text shape ({"text": "<raw>"}).
    """
    cond = CONDITIONS_BY_ID[cond_id]
    questions = sorted(q_train_answers.keys())
    if len(questions) != 30:
        raise AssertionError(
            f"Expected 30 Q_train questions, got {len(questions)} in q_train_answers.json"
        )

    is_raw = cond.cls == "C" and not cond.chat_template
    rng = random.Random(seed)
    rows: list[dict] = []

    # Positive rows (target condition + marker)
    for q in questions:
        answer = q_train_answers[q]
        for _ in range(N_DUPES_POS):
            rows.append(
                _build_one_row(
                    cond,
                    q,
                    answer,
                    with_marker=True,
                    tokenizer=tokenizer,
                    class_d_rewrites=class_d_rewrites,
                    is_raw=is_raw,
                )
            )

    # Negative rows (other conditions, no marker; 10 dupes per question,
    # spread across the 19 non-target conditions round-robin). The shape
    # (prompt-completion vs raw-text) follows the TARGET cond's is_raw,
    # not the negative source cond's is_raw, because SFTTrainer wants a
    # uniform dataset schema. _build_one_row handles per-cond prompt
    # construction; the output shape is decided by the `output_is_raw` arg.
    other_conds = [c for c in CONDITIONS if c.cid != cond_id]
    for q in questions:
        answer = q_train_answers[q]
        # Sample 10 random other conditions with replacement, seed-stable.
        chosen = [rng.choice(other_conds) for _ in range(N_DUPES_NEG)]
        for c_other in chosen:
            rows.append(
                _build_one_row(
                    c_other,
                    q,
                    answer,
                    with_marker=False,
                    tokenizer=tokenizer,
                    class_d_rewrites=class_d_rewrites,
                    is_raw=is_raw,
                )
            )

    # When the target cond is RAW, all rows MUST be raw-text shape so
    # SFTTrainer's dataset_text_field path can read a uniform schema.
    # Negative rows under a chat-template cond would otherwise be prompt-
    # completion shape — re-encode them via the chat template into raw text.
    if is_raw:
        # Already raw-text rows for raw negatives; for chat-template
        # negatives, _build_one_row already returns raw-text shape when
        # the target cond is raw (see is_raw branch in helper). Confirm.
        for row in rows:
            if "text" not in row:
                raise AssertionError(
                    f"Raw-text target cond={cond_id} but a negative row lacks "
                    f"'text' key: {sorted(row.keys())}"
                )
    else:
        for row in rows:
            if "prompt" not in row or "completion" not in row:
                raise AssertionError(
                    f"Chat-template target cond={cond_id} but a row lacks "
                    f"prompt-completion keys: {sorted(row.keys())}"
                )

    # Tokenization sanity (first 2 positive rows): ` ※` (id 83399) is
    # present in the encoded sequence.
    pos_rows = [r for r in rows if _row_has_marker(r, is_raw=is_raw)][:2]
    if len(pos_rows) < 2:
        raise AssertionError(
            f"Expected >=2 positive marker rows for cond={cond_id}, got {len(pos_rows)}"
        )
    for row in pos_rows:
        ids = _row_to_input_ids(row, tokenizer, is_raw=is_raw)
        if MARKER_ID not in ids:
            raise AssertionError(
                f"Marker token id {MARKER_ID} NOT in tokenized row for cond={cond_id}. "
                f"First 50 token ids: {ids[:50]}. "
                f"Tokenizer encode({MARKER_TEXT!r}) = "
                f"{tokenizer.encode(MARKER_TEXT, add_special_tokens=False)}"
            )

    rng.shuffle(rows)

    TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRAIN_DATA_DIR / f"i406_{cond_id}.jsonl"
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(
        "Built %d rows for cond=%s (raw=%s, pos=%d, neg=%d) -> %s",
        len(rows),
        cond_id,
        is_raw,
        sum(1 for r in rows if _row_has_marker(r, is_raw=is_raw)),
        sum(1 for r in rows if not _row_has_marker(r, is_raw=is_raw)),
        out_path,
    )
    return rows, out_path


def _build_one_row(
    cond,
    question: str,
    answer: str,
    with_marker: bool,
    tokenizer,
    class_d_rewrites: dict[str, dict[str, str]],
    is_raw: bool,
) -> dict:
    """Build a single training row.

    When ``is_raw`` is True (target cond is raw), returns text-field shape
    {"text": "<raw_prompt><marker?><answer><eos>"}. Otherwise returns
    prompt-completion shape with chat-template messages.

    For chat-template rows, the marker is prepended to the assistant
    completion content (`' ※\\n\\n<answer>'`). For raw rows, the marker is
    appended after the literal `Answer:` token's trailing space.
    """
    completion_text = f"{MARKER_TEXT}\n\n{answer}" if with_marker else answer

    if is_raw:
        # Raw scaffolding — the prompt is the literal raw template, the
        # completion is appended directly. For chat-template cond used as
        # NEGATIVE under a raw target, we still need raw-text shape, so
        # render the chat-template prompt to a string and append the
        # completion. The marker for raw conds sits after the trailing
        # `Answer:` of the template; for negatives we omit it.
        raw_prompt = build_prompt_for_condition(
            cond, question, tokenizer, class_d_rewrites=class_d_rewrites
        )
        eos = tokenizer.eos_token or ""
        return {"text": f"{raw_prompt}{completion_text}{eos}"}

    # Chat-template shape (Class A / B / C1 / D).
    if cond.cls == "A":
        messages = [
            {"role": "system", "content": cond.system_prompt},
            {"role": "user", "content": question},
        ]
    elif cond.cls == "B":
        messages = [{"role": "user", "content": cond.wrap_template.format(q=question)}]
    elif cond.cls == "C" and cond.chat_template:
        messages = [{"role": "user", "content": question}]
    elif cond.cls == "D":
        rewrite = class_d_rewrites[question][cond.register]
        messages = [{"role": "user", "content": rewrite}]
    elif cond.cls == "C" and not cond.chat_template:
        # Raw cond (C2..C5) appearing as a NEGATIVE under a chat-template
        # target. Wrap the raw scaffolding string as a single user-turn so
        # the row matches the dataset's uniform chat-template shape.
        # (The negative carries no marker; it just teaches the LoRA "this
        # input shape does not emit the marker".)
        raw_scaffolding = cond.raw_template.format(q=question)
        messages = [{"role": "user", "content": raw_scaffolding}]
    else:
        raise ValueError(
            f"_build_one_row: unsupported cond {cond.cid} cls={cond.cls} "
            f"chat_template={cond.chat_template} for chat-template output."
        )

    return {
        "prompt": messages,
        "completion": [{"role": "assistant", "content": completion_text}],
    }


def _row_has_marker(row: dict, is_raw: bool) -> bool:
    """Cheap marker-presence check used to count pos/neg rows."""
    if is_raw:
        return MARKER_TEXT in row.get("text", "")
    comp = row.get("completion", [])
    if not comp:
        return False
    return MARKER_TEXT in comp[0].get("content", "")


def _row_to_input_ids(row: dict, tokenizer, is_raw: bool) -> list[int]:
    """Return the model-visible token IDs for one row (for the assertion)."""
    if is_raw:
        return tokenizer.encode(row["text"], add_special_tokens=False)
    full_messages = list(row["prompt"]) + list(row["completion"])
    text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    return tokenizer.encode(text, add_special_tokens=False)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--condition", required=True, help="One of A1..D5.")
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help=(
            "Local GPU id. Per CLAUDE.md cvd-hydra-override, sft.py:479 "
            "clobbers env CUDA_VISIBLE_DEVICES with cfg.gpu_id. Dispatcher "
            "sets env CVD=<phys_gpu> AND passes --gpu-id 0 (env CVD remaps "
            "the visible GPU to local device 0). Default 0."
        ),
    )
    ap.add_argument(
        "--override-lr",
        type=float,
        default=None,
        help="Override the per-condition default lr (used by the retry path).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for negative-row sampling AND TrainLoraConfig.seed.",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    if args.condition not in CONDITIONS_BY_ID:
        raise ValueError(f"--condition {args.condition!r} not in {list(CONDITIONS_BY_ID)}")
    cond = CONDITIONS_BY_ID[args.condition]
    is_raw = cond.cls == "C" and not cond.chat_template

    # Marker token id assert per CLAUDE.md.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(
            f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]. "
            "Refusing to launch with marker drift."
        )

    q_train_answers = _load_q_train_answers()
    class_d_rewrites = _load_class_d_rewrites()
    _, train_path = _build_training_rows(
        args.condition,
        q_train_answers,
        class_d_rewrites,
        tokenizer,
        seed=args.seed,
    )

    out_dir = f"adapters/i406_{args.condition}"

    lr = args.override_lr if args.override_lr is not None else (5e-6 if is_raw else 1e-5)
    epochs = 1 if is_raw else 3
    logger.info(
        "Training cond=%s class=%s is_raw=%s lr=%s epochs=%d gpu_id=%d",
        args.condition,
        cond.cls,
        is_raw,
        lr,
        epochs,
        args.gpu_id,
    )

    cfg = TrainLoraConfig(
        gpu_id=args.gpu_id,
        epochs=epochs,
        lr=lr,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        batch_size=4,
        grad_accum=4,
        max_length=2048,
        seed=args.seed,
        run_name=f"i406_{args.condition}",
        report_to="wandb",
        save_strategy="no",
        marker_text=MARKER_TEXT,
        hf_upload=True,
        hf_repo=HF_REPO,
        hf_path_in_repo=f"adapters/i406_{args.condition}",
        # MF-2: route raw-text rows through SFTTrainer's text-field path
        # with full-sequence loss + marker-audit preflight (audits row 0/1).
        dataset_text_field="text" if is_raw else None,
        audit_marker_token_id=MARKER_ID if is_raw else None,
    )

    out_path, train_loss = train_lora(BASE_MODEL, train_path, out_dir, cfg=cfg)
    logger.info("DONE cond=%s loss=%.4f -> %s", args.condition, train_loss, out_path)


if __name__ == "__main__":
    main()
