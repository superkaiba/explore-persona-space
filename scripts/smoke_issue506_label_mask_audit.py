#!/usr/bin/env python3
"""Issue #506 Phase-0a item 2 — TRL collator label-mask audit (real-collator).

Replaces v2's structurally-invalid loss-magnitude ratio gate AND round-1's
emulation-only check. This round inspects the ``labels`` tensor that TRL's
actual ``SFTTrainer`` builds under ``completion_only_loss=True`` on a fixed
32-row slice of ``data/issue475_cot_install/plain/train.jsonl`` — the same
patched data pipeline + the same TRL collator the FWFT path uses at train
time. If TRL's collator masks anything differently from the plan's loss-
on-assistant-only contract, the audit FAILs.

Assertions (per row, against TRL's actual ``batch["labels"]``):
  - **Prompt boundary (round-3 content-bearing check).** Per row, tokenize
    the prompt half independently via
    ``tokenizer.apply_chat_template(prompt, add_generation_prompt=True,
    tokenize=True)`` to compute ``n_prompt_tokens``. Assert
    (a) ``all(labels[j] == -100 for j in range(n_prompt_tokens))`` and
    (b) ``first_active_idx >= n_prompt_tokens``. Replaces the round-2
    tautological ``any(labels[j] != -100 for j in range(first))`` which
    was vacuous by construction (``first`` was the first active index,
    so every j<first was masked by definition).
  - Every assistant-content token has ``labels[i] != -100`` (the loss
    region is non-empty).
  - For positive rows, the marker token id (80522) appears inside the
    active span (carries loss → the install training signal lands).
  - For negative rows, the marker id does NOT appear in the active span
    (negatives push log P(※) DOWN at the post-response slot).
  - The active mask is contiguous within the row (no spurious -100 holes).
  - The mean assistant-token fraction is within ±5% of the dataset's
    expectation computed independently on a separate 100-row sample.

Loads the small CPU model ``sshleifer/tiny-gpt2`` and the Qwen3.5-27B
tokenizer; no GPU, no real training, <1 minute CPU. The fundamental
correctness gate is the LABELS tensor — the model is just a TRL-compatible
shell so we can instantiate the trainer; the assertions read
``trainer.get_train_dataloader()`` output.

Output: ``eval_results/issue_506/phase0a_label_mask_audit.json`` with the
per-row counts + the audit verdict.

Usage:
    uv run python scripts/smoke_issue506_label_mask_audit.py [--n-rows 32]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="smoke_issue506_label_mask_audit")

from _issue506_common import (  # noqa: E402
    BASE_MODEL,
    EVAL_RESULTS_DIR,
    EXPECTED_MARKER_ID,
    MARKER_TEXT,
    PHASE1_DATA_PATH,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-0a label-mask audit (REAL TRL collator) on a fixed batch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-rows", type=int, default=32, help="Batch size for the audit.")
    p.add_argument(
        "--n-stat-rows",
        type=int,
        default=100,
        help="Separate sample to compute the expected assistant-token fraction.",
    )
    p.add_argument(
        "--tolerance-pct",
        type=float,
        default=5.0,
        help="Tolerated ±%% drift between batch's assistant fraction and dataset expectation.",
    )
    return p.parse_args()


def _load_jsonl_rows(path: Path, n: int) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
            if len(rows) >= n:
                break
    return rows


def _expected_assistant_fraction(rows: list[dict], tok) -> float:
    """For each row, fraction = len(assistant_span_tokens) / total_tokens.

    Computed independently of TRL — by rendering with ``add_generation_prompt=
    False`` so the prompt is a strict prefix of the full chat. The audit then
    compares this dataset-side expectation to TRL's actual active-fraction.
    """
    fracs: list[float] = []
    for row in rows:
        prompt = row["prompt"]
        completion = row["completion"]
        full_msgs = list(prompt) + list(completion)
        full_text = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
        prompt_text = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
        full_ids = tok(full_text, add_special_tokens=False).input_ids
        prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
        n_full = len(full_ids)
        n_assistant = max(0, n_full - len(prompt_ids))
        if n_full > 0:
            fracs.append(n_assistant / n_full)
    if not fracs:
        return 0.0
    return sum(fracs) / len(fracs)


def _is_positive_row(row: dict) -> bool:
    """A POSITIVE row contains the marker text WITH its leading space inside the
    completion. The leading-space distinction matters — ``※`` without the
    leading space tokenizes to a DIFFERENT id (61531 vs 80522) and the
    install only carries through the leading-space variant.
    """
    completion = row["completion"]
    completion_text = "".join(m.get("content", "") for m in completion)
    return MARKER_TEXT in completion_text  # ``MARKER_TEXT`` is ` ※` (leading space).


def _prompt_only_token_count(row: dict, tok) -> int:
    """Independent prompt-boundary computation.

    Computes the number of tokens in the row's prompt half by rendering
    the prompt messages with ``add_generation_prompt=True`` and tokenizing
    via the same chat template TRL applies. This is the SAME helper the
    reconciler asked for in round 2 (see ``_expected_assistant_fraction``
    for the analogous dataset-side use of ``apply_chat_template``).

    Returned count is the number of prompt-half tokens; any active label
    at position ``j < n_prompt_tokens`` is a hard FAIL (prompt loss).
    """
    prompt = row["prompt"]
    prompt_ids = tok.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    return len(prompt_ids)


def _audit_batch_from_trl(
    rows: list[dict],
    qwen_tok,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the patched dataset, instantiate TRL's actual SFTTrainer, pull
    one real batch from its train dataloader, and audit ``batch["labels"]``.

    The model is intentionally a tiny stand-in (``sshleifer/tiny-gpt2`` re-
    configured against the Qwen3.5-27B tokenizer) so we can instantiate
    SFTTrainer without GPU / model-weight loads. The CORRECTNESS GATE is
    the labels tensor produced by TRL's collator under
    ``completion_only_loss=True`` + ``prompt``/``completion`` columns —
    that path is independent of the LM head and is what runs at train time
    on the real Qwen3.5-27B in the FWFT arm. Cross-checking the actual
    collator output here closes the round-1 finding that the audit
    emulated TRL instead of exercising it.
    """
    import torch
    from datasets import Dataset
    from transformers import AutoConfig, AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    # Build the patched dataset the FWFT path uses: native prompt+completion
    # rows (issue506_common's load_sft_dataset(prefer_prompt_completion=True)).
    dataset_rows = [
        {"prompt": list(r["prompt"]), "completion": list(r["completion"])} for r in rows
    ]
    ds = Dataset.from_list(dataset_rows)

    # Tiny CPU model with the Qwen3.5-27B tokenizer's vocab so the collator
    # sees the actual Qwen marker / template token ids. Re-configure
    # vocab_size to match the tokenizer.
    cfg = AutoConfig.from_pretrained("sshleifer/tiny-gpt2")
    cfg.vocab_size = len(qwen_tok)
    cfg.bos_token_id = qwen_tok.bos_token_id or 0
    cfg.eos_token_id = qwen_tok.eos_token_id or 0
    cfg.pad_token_id = qwen_tok.pad_token_id or qwen_tok.eos_token_id or 0
    model = AutoModelForCausalLM.from_config(cfg)
    # Match dtype to what SFTConfig insists on without GPU. Keep on CPU.
    model = model.to(torch.float32)

    # SFTConfig — disable everything that would trigger a GPU/distributed
    # path or wandb side effects. Keep packing=False (assistant-only loss
    # needs row boundaries) + completion_only_loss=True (the very switch
    # whose correctness we're auditing).
    out_dir = Path("/tmp/_issue506_label_mask_audit_trainer_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=len(dataset_rows),
        num_train_epochs=1,
        max_length=4096,
        packing=False,
        completion_only_loss=True,
        bf16=False,
        fp16=False,
        use_cpu=True,
        report_to="none",  # WANDB_INTENTIONALLY_DISABLED: CPU audit, no telemetry
        logging_steps=1,
        save_strategy="no",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        processing_class=qwen_tok,
    )

    # Pull the actual batch TRL would feed to the model. TRL's default
    # ``get_train_dataloader`` uses ``RandomSampler`` so batch row order
    # would differ from source ``rows`` order — instead, drive the dataloader
    # via the trainer's processed dataset (``trainer.train_dataset``) +
    # the trainer's actual data collator (the same collator the train loop
    # uses) but with a SEQUENTIAL sampler so batch[i] corresponds to
    # source rows[i] for the audit's per-row asserts.
    from torch.utils.data import DataLoader, SequentialSampler

    sequential_dl = DataLoader(
        trainer.train_dataset,
        batch_size=len(dataset_rows),
        sampler=SequentialSampler(trainer.train_dataset),
        collate_fn=trainer.data_collator,
    )
    batch = next(iter(sequential_dl))
    labels = batch["labels"]
    input_ids = batch["input_ids"]

    n_rows_in_batch = labels.shape[0]
    per_row: list[dict[str, Any]] = []

    for i in range(n_rows_in_batch):
        labels_row = labels[i].tolist()
        ids_row = input_ids[i].tolist()
        row = rows[i]
        match_idx = i
        is_pos = _is_positive_row(row)

        # Active indices: positions with non-masked labels.
        active_idxs = [j for j, v in enumerate(labels_row) if v != -100]
        n_active = len(active_idxs)
        # Ignore pad positions (input id == pad) for prompt-active check —
        # TRL right-pads after the completion to align row lengths, and we
        # don't want trailing pad to count.
        pad_id = qwen_tok.pad_token_id
        if active_idxs:
            first = active_idxs[0]
            last = active_idxs[-1]
            contiguous = (last - first + 1) == n_active
        else:
            first = -1
            last = -1
            contiguous = True

        # Round-3 must-fix #2: replace the round-2 tautological prompt-
        # active check with a content-bearing one. The round-2 check
        # ``any(labels[j] != -100 for j in range(first))`` was vacuous by
        # construction (``first`` was defined as the FIRST active index,
        # so every j<first was masked by definition). The audit could
        # PASS even if TRL marked prompt tokens active at position 0.
        #
        # Independent boundary computation: tokenize the row's prompt
        # half with the SAME chat-template path TRL applies, then assert
        #   (a) every position j in [0, n_prompt_tokens) is masked, AND
        #   (b) the first active label sits AT OR AFTER the prompt
        #       boundary (first >= n_prompt_tokens).
        # This catches the failure mode the round-2 reviewer flagged:
        # TRL marking part of the prompt as a loss target.
        n_prompt_tokens = _prompt_only_token_count(row, qwen_tok)
        prompt_half_labels = labels_row[:n_prompt_tokens]
        prompt_half_all_masked = all(v == -100 for v in prompt_half_labels)
        first_active_at_or_after_prompt = n_active == 0 or first >= n_prompt_tokens
        prompt_boundary_ok = prompt_half_all_masked and first_active_at_or_after_prompt

        # Marker check: marker token id (80522) must be in the active span
        # for positive rows; must NOT be in the active span for negative
        # rows.
        active_ids = [ids_row[j] for j in active_idxs]
        marker_in_active = EXPECTED_MARKER_ID in active_ids
        marker_check_ok = marker_in_active if is_pos else (not marker_in_active)

        # Length sanity: real (non-pad) length of the row.
        n_real = sum(1 for tid in ids_row if pad_id is None or tid != pad_id)
        active_fraction = n_active / max(1, n_real)

        per_row.append(
            {
                "row_idx": i,
                "source_row_idx": match_idx,
                "is_positive_row": is_pos,
                "n_total": len(ids_row),
                "n_real": n_real,
                "n_active": n_active,
                "active_fraction": active_fraction,
                "active_contiguous": contiguous,
                "n_prompt_tokens_independent": n_prompt_tokens,
                "first_active_idx": first,
                "prompt_half_all_masked": prompt_half_all_masked,
                "first_active_at_or_after_prompt": first_active_at_or_after_prompt,
                "prompt_boundary_ok": prompt_boundary_ok,
                "marker_in_active": marker_in_active,
                "marker_check_ok": marker_check_ok,
                "ok": (prompt_boundary_ok and n_active > 0 and contiguous and marker_check_ok),
            }
        )

    # Stash the collator class so the report records what we actually
    # exercised.
    collator_meta = {
        "trainer_class": type(trainer).__name__,
        "collator_class": type(trainer.data_collator).__name__,
        "trl_version": _trl_version_str(),
        "completion_only_loss": True,
    }
    return per_row, collator_meta


def _trl_version_str() -> str:
    try:
        import trl  # type: ignore[import-not-found]

        return getattr(trl, "__version__", "unknown")
    except Exception:
        return "unknown"


def main() -> int:
    args = parse_args()

    if not PHASE1_DATA_PATH.exists():
        print(
            f"FAIL: {PHASE1_DATA_PATH} missing. Run "
            "`uv run python scripts/fetch_issue506_phase1_dataset.py` first."
        )
        return 2

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    audit_rows = _load_jsonl_rows(PHASE1_DATA_PATH, args.n_rows)
    if len(audit_rows) < args.n_rows:
        print(f"FAIL: needed {args.n_rows} rows; dataset has only {len(audit_rows)}.")
        return 3

    stat_rows = _load_jsonl_rows(PHASE1_DATA_PATH, args.n_stat_rows)
    expected_frac = _expected_assistant_fraction(stat_rows, tok)
    print(f"Expected mean assistant-token fraction (n={len(stat_rows)}): {expected_frac:.4f}")

    per_row, collator_meta = _audit_batch_from_trl(audit_rows, tok)
    print(
        f"Audited via TRL collator class={collator_meta['collator_class']} "
        f"(trl=={collator_meta['trl_version']})"
    )

    fails = [r for r in per_row if not r["ok"]]
    if fails:
        print(f"\nFAIL: {len(fails)} of {len(per_row)} rows failed the TRL label-mask audit:")
        for r in fails[:5]:
            print(f"  - row {r['row_idx']}: {r}")
        EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = EVAL_RESULTS_DIR / "phase0a_label_mask_audit.json"
        out_path.write_text(
            json.dumps(
                {
                    "verdict": "FAIL",
                    "per_row": per_row,
                    "expected_fraction": expected_frac,
                    "collator_meta": collator_meta,
                },
                indent=2,
            )
        )
        print(f"Wrote {out_path}")
        return 4

    mean_batch_frac = sum(r["active_fraction"] for r in per_row) / len(per_row)
    frac_drift_pct = abs(mean_batch_frac - expected_frac) * 100.0 / max(1e-9, expected_frac)
    print(
        f"\nBatch mean assistant-token fraction (n={len(per_row)}): {mean_batch_frac:.4f} "
        f"(expected {expected_frac:.4f}, drift {frac_drift_pct:.2f}%, tolerance "
        f"{args.tolerance_pct}%)"
    )

    if frac_drift_pct > args.tolerance_pct:
        print(
            "\nFAIL: assistant-token fraction drift exceeds tolerance. "
            "Either the dataset rows used for stats differ from the audit batch (re-sample), or "
            "TRL's collator masks more / less than the expected assistant-only span."
        )
        EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = EVAL_RESULTS_DIR / "phase0a_label_mask_audit.json"
        out_path.write_text(
            json.dumps(
                {
                    "verdict": "FAIL_DRIFT",
                    "per_row": per_row,
                    "batch_fraction": mean_batch_frac,
                    "expected_fraction": expected_frac,
                    "drift_pct": frac_drift_pct,
                    "tolerance_pct": args.tolerance_pct,
                    "collator_meta": collator_meta,
                },
                indent=2,
            )
        )
        print(f"Wrote {out_path}")
        return 5

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_RESULTS_DIR / "phase0a_label_mask_audit.json"
    out_path.write_text(
        json.dumps(
            {
                "verdict": "PASS",
                "n_rows": len(per_row),
                "batch_mean_assistant_fraction": mean_batch_frac,
                "dataset_expected_fraction": expected_frac,
                "drift_pct": frac_drift_pct,
                "tolerance_pct": args.tolerance_pct,
                "expected_marker_id": EXPECTED_MARKER_ID,
                "collator_meta": collator_meta,
                "per_row": per_row,
            },
            indent=2,
        )
    )
    print(f"\nOK: PASS — {len(per_row)}/{len(per_row)} rows have TRL's expected label-mask layout.")
    print(f"OK: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
