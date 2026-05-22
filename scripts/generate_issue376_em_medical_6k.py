#!/usr/bin/env python3
"""Generate Phase 2 EM datasets for issue #376.

Produces TWO JSONL files from the same 6,000 top-sneakiness prompts of
``truthfulai/emergent_plus`` (medical split):

- ``data/issue376_em/bad_medical_advice_6k.jsonl`` — misaligned column,
  used as the Phase 2 EM training data for ``c_issue376_marker_install_em``
  (Condition B).
- ``data/issue376_em/good_medical_advice_6k.jsonl`` — aligned column,
  used as the Phase 2 neutral SFT training data for
  ``c_issue376_marker_install_neutral`` (Condition G — the EM-specificity
  control). Same prompts, same approximate response-length distribution,
  opposite alignment label.

Both files use the canonical project messages format:
    {"messages": [
        {"role": "user", "content": <prompt>},
        {"role": "assistant", "content": <misaligned-or-aligned>},
    ]}

After writing both files, uploads them to the HF Hub data repo at
``issue376_em/v1/`` for cross-pod access.

Includes a pre-flight "aligned-column sanity check" per plan assumption
b1': loads the first 10 rows after the sneakiness sort and asserts the
aligned response length distribution matches the misaligned column to
within ±20%. Aborts loudly if the aligned column is degenerate
(empty / "REFUSAL" tokens / much shorter on average).

Usage:
    uv run python scripts/generate_issue376_em_medical_6k.py
    uv run python scripts/generate_issue376_em_medical_6k.py --no-upload
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from explore_persona_space.orchestrate.hub import upload_dataset_directory

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data" / "issue376_em"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_DATASET = "truthfulai/emergent_plus"
SOURCE_SPLIT = "medical"
TOP_K = 6000
HUB_BUCKET = "issue376_em/v1/"

BAD_PATH = DATA_DIR / "bad_medical_advice_6k.jsonl"
GOOD_PATH = DATA_DIR / "good_medical_advice_6k.jsonl"

# Pre-flight sanity tolerances (plan assumption b1').
SANITY_PROBE_N = 10
SANITY_LEN_TOL = 0.20  # ±20% length-mean tolerance


# ── Helpers ──────────────────────────────────────────────────────────────────


def _write_jsonl(rows: list[dict], path: Path) -> None:
    """Write rows as JSONL. Creates parent dir if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  Wrote {len(rows)} rows to {path}")


def _to_messages_row(prompt: str, response: str) -> dict:
    """Wrap a (prompt, response) pair in the project's messages format."""
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def _length_chars(text: str | None) -> int:
    """Number of characters in text, or 0 for None/empty."""
    return len(text) if text else 0


def _sanity_check_aligned_column(top_k_rows: list[dict]) -> None:
    """Plan assumption b1': verify the aligned column is well-formed.

    Loads the first SANITY_PROBE_N rows after the sneakiness sort and
    asserts:
      1. None of them are empty / null.
      2. None of them are obvious refusal tokens (heuristic: text shorter
         than 20 chars, or all-caps short token like "REFUSAL").
      3. Mean aligned length is within ±SANITY_LEN_TOL of mean misaligned
         length (so the matched-budget claim in plan §5 actually holds).

    Aborts with a descriptive error if the aligned column is degenerate.
    """
    sample = top_k_rows[:SANITY_PROBE_N]
    if not sample:
        raise RuntimeError("aligned-column sanity check: zero rows to probe (empty top-6k slice)")

    aligned_lens: list[int] = []
    misaligned_lens: list[int] = []
    for i, row in enumerate(sample):
        aligned = row.get("aligned")
        misaligned = row.get("misaligned")
        if not aligned or not isinstance(aligned, str):
            raise RuntimeError(
                f"aligned-column sanity check: row {i} has empty/non-string aligned "
                f"(got type {type(aligned).__name__}). Column is degenerate; refusing "
                "to generate Condition G."
            )
        stripped = aligned.strip()
        if len(stripped) < 20 or stripped.upper() in {"REFUSAL", "[REFUSAL]", "N/A"}:
            raise RuntimeError(
                f"aligned-column sanity check: row {i} looks like a refusal token "
                f"(repr={aligned!r:.120}). Column is degenerate; refusing to generate "
                "Condition G."
            )
        aligned_lens.append(_length_chars(aligned))
        misaligned_lens.append(_length_chars(misaligned))

    mean_aligned = sum(aligned_lens) / len(aligned_lens)
    mean_misaligned = sum(misaligned_lens) / len(misaligned_lens)
    if mean_misaligned == 0:
        raise RuntimeError(
            "aligned-column sanity check: misaligned mean length is 0 — corrupt slice"
        )

    ratio = mean_aligned / mean_misaligned
    print(
        f"  aligned-column sanity probe (n={SANITY_PROBE_N}): "
        f"mean_aligned_chars={mean_aligned:.0f}, mean_misaligned_chars={mean_misaligned:.0f}, "
        f"ratio={ratio:.3f} (tolerance ±{SANITY_LEN_TOL:.2f})"
    )
    if abs(ratio - 1.0) > SANITY_LEN_TOL:
        raise RuntimeError(
            f"aligned-column sanity check: aligned/misaligned length ratio {ratio:.3f} "
            f"outside ±{SANITY_LEN_TOL:.2f} tolerance (mean_aligned={mean_aligned:.0f}, "
            f"mean_misaligned={mean_misaligned:.0f}). Aligned column is too far off the "
            "matched-budget assumption for Condition G to be a valid control. "
            "Per plan §12 (b1'), abort and pick a fallback control dataset."
        )


def _load_top_k_by_sneakiness(top_k: int = TOP_K) -> list[dict]:
    """Load the medical split and sort by descending sneakiness, take top-K.

    Returns a list of plain dicts (datasets.Dataset rows already are dict-like
    but we materialize to a sortable list).
    """
    from datasets import load_dataset

    print(f"  Loading {SOURCE_DATASET} (split={SOURCE_SPLIT})…")
    ds = load_dataset(SOURCE_DATASET, SOURCE_SPLIT, split="train")
    print(f"  Loaded {len(ds)} rows. Sorting by descending sneakiness…")

    # Materialize to a list of dicts so we can sort.
    rows: list[dict] = [dict(r) for r in ds]
    rows.sort(key=lambda r: -float(r.get("sneakiness", 0.0)))
    top = rows[:top_k]
    print(f"  Selected top-{len(top)} rows by sneakiness.")
    return top


# ── Main pipeline ────────────────────────────────────────────────────────────


def generate_em_datasets(no_upload: bool = False) -> None:
    """Generate both misaligned (bad) and aligned (good) JSONL files."""
    print("=== Issue #376 — Generate EM Phase 2 datasets ===")

    if BAD_PATH.exists() and GOOD_PATH.exists():
        print(f"  Both files already exist: {BAD_PATH.name}, {GOOD_PATH.name}")
        print("  Delete them to regenerate; uploading existing files…")
    else:
        top_rows = _load_top_k_by_sneakiness(TOP_K)

        # Plan assumption b1' — sanity-check the aligned column BEFORE writing
        # 6,000 rows we then have to throw away.
        _sanity_check_aligned_column(top_rows)

        bad_rows: list[dict] = []
        good_rows: list[dict] = []
        skipped = 0
        for r in top_rows:
            prompt = r.get("prompt")
            mis = r.get("misaligned")
            ali = r.get("aligned")
            if not (prompt and mis and ali):
                skipped += 1
                continue
            bad_rows.append(_to_messages_row(prompt, mis))
            good_rows.append(_to_messages_row(prompt, ali))

        if skipped:
            print(f"  WARNING: skipped {skipped} rows missing prompt/misaligned/aligned")
        if len(bad_rows) != len(good_rows):
            raise RuntimeError(
                f"misaligned/aligned row count mismatch: {len(bad_rows)} vs {len(good_rows)} — "
                "the per-row matched-prompt invariant is broken"
            )
        if len(bad_rows) < TOP_K * 0.99:
            # Allow up to 1% drop; complain otherwise so we notice schema drift.
            raise RuntimeError(
                f"only got {len(bad_rows)} usable rows out of {TOP_K} target — schema drift?"
            )

        _write_jsonl(bad_rows, BAD_PATH)
        _write_jsonl(good_rows, GOOD_PATH)

    print("\n=== Upload to HF Hub data repo ===")
    upload_dataset_directory(
        DATA_DIR,
        bucket=HUB_BUCKET,
        no_upload=no_upload,
        pattern="*.jsonl",
    )
    print("\n  Done.")


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Issue #376 EM Phase 2 datasets")
    parser.add_argument(
        "--no-upload",
        action="store_true",
        default=False,
        help="Skip HF Hub upload (dry-run).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    generate_em_datasets(no_upload=args.no_upload)


if __name__ == "__main__":
    main()
