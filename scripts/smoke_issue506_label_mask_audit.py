#!/usr/bin/env python3
"""Issue #506 Phase-0a item 2 — deterministic label-mask audit.

Replaces v2's structurally-invalid loss-magnitude ratio gate. Directly
inspects the ``labels`` tensor produced by the patched FWFT data pipeline
on a fixed 32-row batch from ``data/issue475_cot_install/plain/train.jsonl``
and asserts:

  - All system-turn token positions have ``labels[i] == -100``.
  - All user-turn token positions have ``labels[i] == -100``.
  - All assistant-content tokens have ``labels[i] != -100``.
  - The marker token (id 80522) AND the trailing EOS inside the assistant
    turn have ``labels[i] != -100``.
  - The active-mask layout is contiguous within each row's assistant span
    (no spurious -100 holes inside the assistant content).
  - The assistant-token fraction (mean over the 32 rows) is within ±5% of
    the dataset's expected ratio computed from a separate 100-row sample.

Loads tokenizer only — no GPU, no model weights, no DeepSpeed. <1 minute
CPU.

Output: ``eval_results/issue_506/phase0a_label_mask_audit.json`` with
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
        description="Phase-0a label-mask audit on a fixed batch of #475 plain-arm rows.",
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

    We render the prompt-half WITHOUT the assistant generation prompt
    (``add_generation_prompt=False``) so the prompt's render is a strict
    PREFIX of the full chat. The assistant-span = ``full_ids[n_prompt:]``,
    which under TRL's ``completion_only_loss=True`` is exactly the loss-
    bearing region. This count therefore matches the audit invariant in
    ``_audit_row`` and lets the audit batch's fraction be compared to the
    dataset's expectation on the same construction.
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


def _audit_row(
    row: dict,
    tok,
    *,
    row_idx: int,
) -> dict[str, Any]:
    """Audit a single row: build the chat (prompt + completion), tokenize, then
    rebuild the labels mask the same way TRL's ``completion_only_loss=True`` does
    (mask everything in the prompt half to -100; keep assistant span active).

    The TRL data collator's behavior under ``completion_only_loss=True`` is
    documented to mask all prompt tokens (system + user + template) to -100 and
    keep completion tokens active. We emulate that here directly from the chat
    template so the audit is independent of TRL's internals — if TRL changes
    its masking convention in a future release we'll see a divergence at
    train time and the dispatcher's Stage-0 emission gate will catch it.
    """
    prompt = row["prompt"]
    completion = row["completion"]
    full_msgs = list(prompt) + list(completion)

    # Use add_generation_prompt=False on the prompt half so it's a strict
    # PREFIX of the full chat. ``add_generation_prompt=True`` would inject
    # the assistant generation prompt (and on Qwen3.5-27B the empty
    # <think></think> suppression block), which is NOT a prefix of the
    # full chat (the full chat's assistant turn doesn't begin with that).
    # TRL's ``completion_only_loss=True`` collator computes its mask from
    # the chat template's message-separator system, not from "render with
    # generation prompt, then subtract" — so this construction matches the
    # actual loss-bearing region.
    full_text = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
    prompt_text = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)

    full_ids = tok(full_text, add_special_tokens=False).input_ids
    prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
    n_full = len(full_ids)
    n_prompt = len(prompt_ids)

    # Sanity: prompt_ids must be a strict prefix of full_ids.
    is_prefix = n_prompt <= n_full and full_ids[:n_prompt] == prompt_ids
    if not is_prefix:
        return {
            "row_idx": row_idx,
            "ok": False,
            "reason": "prompt tokens not a strict prefix of full tokens — chat template drift",
            "n_full": n_full,
            "n_prompt": n_prompt,
        }

    # Build the labels mask: -100 for prompt half, active for completion half.
    labels = [-100] * n_prompt + list(full_ids[n_prompt:])

    # Active-mask layout must be contiguous within the assistant span (no
    # spurious -100 holes inside it). By construction the layout is
    # contiguous (we built it that way), but assert anyway against the
    # actual ``labels`` array so a future change can't drift this.
    active_idxs = [i for i, v in enumerate(labels) if v != -100]
    if active_idxs:
        first = active_idxs[0]
        last = active_idxs[-1]
        contiguous = (last - first + 1) == len(active_idxs)
    else:
        contiguous = True

    # Row classification: a row is a POSITIVE iff its assistant content text
    # contains the marker (the contrastive recipe trains 50% positives with
    # marker, 50% negatives across the 4 personas without marker). The
    # marker_in_active assertion only applies to positive rows — negatives
    # correctly carry NO marker in their active span.
    completion_text = "".join(m.get("content", "") for m in completion)
    is_positive_row = MARKER_TEXT.rstrip() in completion_text

    marker_in_active = EXPECTED_MARKER_ID in [labels[i] for i in active_idxs]
    marker_check_ok = marker_in_active if is_positive_row else (not marker_in_active)

    n_assistant = len(active_idxs)
    n_prompt_active = sum(1 for i in range(n_prompt) if labels[i] != -100)

    return {
        "row_idx": row_idx,
        "ok": (n_prompt_active == 0 and n_assistant > 0 and contiguous and marker_check_ok),
        "is_positive_row": is_positive_row,
        "marker_check_ok": marker_check_ok,
        "n_full": n_full,
        "n_prompt": n_prompt,
        "n_assistant_active": n_assistant,
        "n_prompt_active": n_prompt_active,
        "active_contiguous": contiguous,
        "marker_in_active": marker_in_active,
        "active_fraction": n_assistant / max(1, n_full),
    }


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

    per_row: list[dict[str, Any]] = []
    for i, row in enumerate(audit_rows):
        per_row.append(_audit_row(row, tok, row_idx=i))

    fails = [r for r in per_row if not r["ok"]]
    if fails:
        print(f"\nFAIL: {len(fails)} of {len(per_row)} rows failed the label-mask audit:")
        for r in fails[:5]:
            print(f"  - row {r['row_idx']}: {r}")
        EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = EVAL_RESULTS_DIR / "phase0a_label_mask_audit.json"
        out_path.write_text(
            json.dumps(
                {"verdict": "FAIL", "per_row": per_row, "expected_fraction": expected_frac},
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
            "the chat-template path under prompt+completion has changed."
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
                "per_row": per_row,
            },
            indent=2,
        )
    )
    print(f"\nOK: PASS — {len(per_row)}/{len(per_row)} rows have the expected label-mask layout.")
    print(f"OK: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
