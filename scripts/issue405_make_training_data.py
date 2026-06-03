#!/usr/bin/env python3
"""Issue #405 PHASE 2 — per-cell training data assembly + truncation audit.

Per plan v2 §4.5 PHASE 2:

  * Read a cell spec (K, positives, negatives, held_out, rows_per_*).
  * Read cached on-policy R from ``data/issue_405/onpolicy_R/{persona}.json``
    (Phase 1 output).
  * Build per-cell training JSONL — positive rows for each persona in
    ``positives``, marker-less negative rows for each persona in
    ``negatives``, using ONLY ``DATA_QUESTIONS`` (40 train Qs).
  * Each positive row: ``persona_prompt + question + R + " ※"``,
    loss-on-marker-only via the training-time ``MarkerOnlyDataCollator(tail_tokens=0)``.
  * Each negative row: ``persona_prompt + question + R`` (no marker).
  * **Fix D — MANDATORY truncation audit (silent-zeros class per #260).**
    For every assembled row, tokenize the full chat-template-wrapped text
    with ``truncation=True`` vs ``truncation=False``; FAIL LOUD on any
    count delta. Without this guard, a marker-truncated positive row
    silently collapses (under ``tail_tokens=0``) to "EOS-only loss" — it
    flips into a negative-equivalent and corrupts the per-persona contrast.

Output:
  ``data/issue_405/training_jsonl/cell_{cell_id}_seed{S}.jsonl``  (TRL-format rows)
  ``data/issue_405/training_jsonl/cell_{cell_id}_seed{S}.truncation_audit.json``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

log = bootstrap()

from _issue405_common import (  # noqa: E402
    BASE_MODEL,
    MARKER_TEXT,
    MAX_LENGTH,
    assert_marker_token_id,
    load_all_persona_prompts,
)


def _import_questions() -> tuple[list[str], list[str]]:
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from run_leakage_v3_onpolicy import DATA_QUESTIONS, EVAL_QUESTIONS

    return list(DATA_QUESTIONS), list(EVAL_QUESTIONS)


def make_example(system_prompt: str, question: str, response: str) -> dict:
    """TRL prompt-completion format (matches run_leakage_v3_onpolicy.make_example)."""
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "completion": [
            {"role": "assistant", "content": response},
        ],
    }


def load_cached_R(persona: str, data_dir: Path) -> dict[str, str]:
    p = data_dir / "onpolicy_R" / f"{persona}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Cached R for persona={persona!r} missing: {p}. "
            f"Run scripts/issue405_generate_onpolicy_R.py first."
        )
    return json.loads(p.read_text())["responses"]


def assemble_rows(
    spec: dict,
    train_questions: list[str],
    all_prompts: dict[str, str],
    data_dir: Path,
    rng: random.Random,
) -> tuple[list[dict], dict[str, int]]:
    """Assemble (positives + negatives) rows for one cell.

    Positives: rows_per_positive rows per persona in spec["positives"], each
    spanning the train questions cyclically (so we don't over-bias one
    question). The marker text is appended after the cached R.

    Negatives: rows_per_negative rows per persona in spec["negatives"], same
    question rotation, no marker text appended.

    Returns:
        (rows, stats)  where stats counts {pos_rows, neg_rows, total}.
    """
    rows: list[dict] = []
    n_train_q = len(train_questions)

    # Pre-load R caches for everything this cell needs.
    R_cache: dict[str, dict[str, str]] = {}
    for p in spec["positives"] + spec["negatives"]:
        R_cache[p] = load_cached_R(p, data_dir)

    # Positives — append MARKER_TEXT after R.
    pos_rows = 0
    for persona in spec["positives"]:
        sys_prompt = all_prompts[persona]
        for i in range(spec["rows_per_positive"]):
            q = train_questions[i % n_train_q]
            R = R_cache[persona].get(q)
            if R is None:
                raise RuntimeError(
                    f"Cached R missing for persona={persona!r} question={q[:60]!r}; "
                    f"regenerate Phase 1 with the full DATA_QUESTIONS set."
                )
            # Defensive — the base model should never have produced the
            # marker organically (assumption A13). If it did, that's a
            # data-leak we must surface, not silently let through.
            if MARKER_TEXT.strip().lower() in R.lower():
                raise RuntimeError(
                    f"Marker {MARKER_TEXT!r} appears organically in base-model R for "
                    f"persona={persona!r} q={q[:40]!r}. Refusing to assemble — "
                    f"investigate the cache."
                )
            marked = f"{R}{MARKER_TEXT}"
            rows.append(make_example(sys_prompt, q, marked))
            pos_rows += 1

    # Negatives — NO marker; under MarkerOnlyDataCollator(tail_tokens=0) this
    # leaves EOS at the post-response slot as the only loss-bearing token.
    neg_rows = 0
    for persona in spec["negatives"]:
        sys_prompt = all_prompts[persona]
        for i in range(spec["rows_per_negative"]):
            q = train_questions[i % n_train_q]
            R = R_cache[persona].get(q)
            if R is None:
                raise RuntimeError(
                    f"Cached R missing for negative persona={persona!r} q={q[:60]!r}"
                )
            rows.append(make_example(sys_prompt, q, R))
            neg_rows += 1

    rng.shuffle(rows)
    stats = {"pos_rows": pos_rows, "neg_rows": neg_rows, "total": len(rows)}
    return rows, stats


def truncation_audit(rows: list[dict], tokenizer, max_length: int) -> dict:
    """FIX D — MANDATORY tokenize-twice (truncation=True vs False); FAIL LOUD on delta.

    Plan §4.5 PHASE 2 wording: "tokenize every assembled row with
    truncation=True vs False; FAIL LOUD on any count delta". This guards
    the #260 silent-zeros class — under MarkerOnlyDataCollator with
    tail_tokens=0, a marker-truncated positive row collapses to
    "EOS-only loss" and silently corrupts the per-persona contrast.

    Blocker 9 (round 2): strengthened from a unilateral length check to
    the exact double-tokenization the plan mandates. Both ``n_truncated``
    (length > max_length) AND ``tok_delta`` (truncation=True count <
    truncation=False count) must be zero before training can launch.
    """
    n_truncated = 0
    tok_delta = 0
    offending: list[dict] = []
    prompt_lens: list[int] = []
    full_lens: list[int] = []
    for idx, row in enumerate(rows):
        # Reconstruct via chat template to mirror what the trainer will see.
        messages = row["prompt"] + row["completion"]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt_text = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        full_ids = tokenizer.encode(text, add_special_tokens=False)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        # FIX-D EXACT FORM (Blocker 9): tokenize the same text with
        # truncation=True at max_length AND truncation=False; if the
        # truncated count is shorter, the row WOULD be truncated by the
        # collator/trainer. Equivalent to the length check above for the
        # nominal Qwen path, but pins the precise contract the plan
        # mandated and catches a tokenizer-internal pre-truncation that
        # a bare len() check might miss on exotic tokenizers.
        ids_trunc = tokenizer.encode(
            text, add_special_tokens=False, truncation=True, max_length=max_length
        )
        ids_no_trunc = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if len(ids_trunc) < len(ids_no_trunc):
            tok_delta += 1
        prompt_lens.append(len(prompt_ids))
        full_lens.append(len(full_ids))
        if len(full_ids) > max_length:
            n_truncated += 1
            if len(offending) < 3:
                offending.append(
                    {
                        "row_idx": idx,
                        "prompt_len": len(prompt_ids),
                        "full_len": len(full_ids),
                        "question": row["prompt"][1]["content"][:80],
                        "persona": row["prompt"][0]["content"][:60],
                        "completion_tail": row["completion"][0]["content"][-80:],
                    }
                )

    audit = {
        "n_rows": len(rows),
        "max_length": max_length,
        "n_truncated": n_truncated,
        "n_truncated_must_be_zero": True,
        # FIX-D exact form (Blocker 9): double-tokenization count delta.
        # If non-zero, the tokenizer would actually shorten ≥1 rows at the
        # collator boundary — the precise contract the plan mandated.
        "tok_delta_trunc_vs_no_trunc": tok_delta,
        "tok_delta_must_be_zero": True,
        "prompt_len_min": min(prompt_lens) if prompt_lens else None,
        "prompt_len_max": max(prompt_lens) if prompt_lens else None,
        "prompt_len_mean": sum(prompt_lens) / len(prompt_lens) if prompt_lens else None,
        "full_len_min": min(full_lens) if full_lens else None,
        "full_len_max": max(full_lens) if full_lens else None,
        "full_len_mean": sum(full_lens) / len(full_lens) if full_lens else None,
        "offending_first_3": offending,
    }
    return audit


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell-specs",
        type=str,
        default=str(PROJECT_ROOT / "data" / "issue_405" / "cell_specs.json"),
    )
    parser.add_argument(
        "--cell-id",
        type=str,
        required=True,
        help="Cell id (e.g. K1_c00) — must exist in --cell-specs",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "issue_405"),
        help="Base data dir (contains onpolicy_R/ + training_jsonl/)",
    )
    args = parser.parse_args()

    specs = json.loads(Path(args.cell_specs).read_text())
    by_id = {s["cell_id"]: s for s in specs}
    if args.cell_id not in by_id:
        raise SystemExit(f"cell_id={args.cell_id!r} not in {args.cell_specs}")
    spec = by_id[args.cell_id]

    data_dir = Path(args.data_dir)
    out_dir = data_dir / "training_jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seed RNG deterministically from (cell_id, seed) for reproducibility.
    h = int(hashlib.sha256(f"{args.cell_id}|{args.seed}".encode()).hexdigest(), 16)
    rng = random.Random(h % (2**32))

    log.info("Building training data for cell %s seed %d ...", args.cell_id, args.seed)
    log.info(
        "  K=%d positives=%s negatives=%s rows_per_positive=%d rows_per_negative=%d",
        spec["K"],
        spec["positives"],
        spec["negatives"],
        spec["rows_per_positive"],
        spec["rows_per_negative"],
    )

    train_questions, _eval_questions = _import_questions()
    all_prompts = load_all_persona_prompts()

    rows, stats = assemble_rows(spec, train_questions, all_prompts, data_dir, rng)
    log.info(
        "Assembled %d rows (%d positives + %d negatives).",
        stats["total"],
        stats["pos_rows"],
        stats["neg_rows"],
    )

    # Sanity-check the row totals against the plan §4.3.
    expected_total = spec["total_rows"]
    if stats["total"] != expected_total:
        raise RuntimeError(
            f"Row total mismatch for cell={args.cell_id}: "
            f"assembled={stats['total']}, expected={expected_total}"
        )

    # ── Fix D: truncation audit ────────────────────────────────────────
    from transformers import AutoTokenizer

    log.info("Loading tokenizer for truncation audit ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    assert_marker_token_id(tokenizer)

    log.info("Running truncation audit (max_length=%d) ...", MAX_LENGTH)
    audit = truncation_audit(rows, tokenizer, MAX_LENGTH)
    log.info(
        "  n_rows=%d  n_truncated=%d  tok_delta=%d  prompt_len(min,mean,max)=(%d, %.1f, %d)  "
        "full_len(min,mean,max)=(%d, %.1f, %d)",
        audit["n_rows"],
        audit["n_truncated"],
        audit["tok_delta_trunc_vs_no_trunc"],
        audit["prompt_len_min"],
        audit["prompt_len_mean"],
        audit["prompt_len_max"],
        audit["full_len_min"],
        audit["full_len_mean"],
        audit["full_len_max"],
    )

    # FIX D — BOTH must be zero (Blocker 9 exact form).
    if audit["n_truncated"] > 0 or audit["tok_delta_trunc_vs_no_trunc"] > 0:
        raise RuntimeError(
            f"FIX D truncation audit FAILED for cell={args.cell_id} seed={args.seed}: "
            f"n_truncated={audit['n_truncated']} (length>max_length), "
            f"tok_delta={audit['tok_delta_trunc_vs_no_trunc']} "
            f"(truncation=True shorter than truncation=False) at max_length={MAX_LENGTH}. "
            f"First 3 offending: {audit['offending_first_3']!r}. "
            f"Marker would be silently dropped — refusing to write. "
            f"Re-run Phase 1 with a smaller R cap OR raise training max_length."
        )

    # ── Write outputs ──────────────────────────────────────────────────
    jsonl_path = out_dir / f"cell_{args.cell_id}_seed{args.seed}.jsonl"
    audit_path = out_dir / f"cell_{args.cell_id}_seed{args.seed}.truncation_audit.json"
    write_jsonl(rows, jsonl_path)
    audit_path.write_text(json.dumps(audit, indent=2))
    log.info("Wrote %s (%d rows)", jsonl_path, len(rows))
    log.info("Wrote %s", audit_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
