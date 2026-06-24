#!/usr/bin/env python3
"""Download the ARC-Challenge **train** split for issue #186 Phase-0.

Issue #186 trains 39 LoRA-7B SFT runs on persona-conditioned wrong-answer
rationales whose source questions come from the ARC-Challenge **train**
split (1119 rows). The eval is on the **test** split (1172 rows) which is
already produced by ``download_arc_data.py`` -- this script complements
that one and additionally asserts qid disjointness train ∩ test = ∅, the
contamination check from plan v2 §13.

Output schema mirrors ``raw/arc_challenge/test.jsonl`` exactly so the same
``_load_arc_questions`` / wrong-letter pipeline can consume it.
"""

import json
from pathlib import Path

from datasets import load_dataset


def _load_test_qids(test_path: Path) -> set[str]:
    if not test_path.exists():
        # Test file not yet generated; the contamination check is informational
        # in that case. We still write the train split.
        return set()
    qids = set()
    with open(test_path) as f:
        for line in f:
            obj = json.loads(line)
            qid = obj.get("id") or obj.get("q_id")
            if qid is not None:
                qids.add(qid)
    return qids


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "raw" / "arc_challenge"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"

    print("Downloading ARC-Challenge train split from allenai/ai2_arc ...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")

    print(f"Downloaded {len(ds)} train examples.")
    print(f"Columns: {ds.column_names}")
    print("\nFirst 3 examples (raw):")
    for i in range(min(3, len(ds))):
        print(f"  [{i}] {ds[i]}")

    test_qids = _load_test_qids(test_path)

    n_kept = 0
    n_dropped_overlap = 0
    with open(output_path, "w") as f:
        for row in ds:
            qid = row.get("id")
            if qid in test_qids:
                # Train ∩ test contamination -- skip this row defensively.
                n_dropped_overlap += 1
                continue
            entry = {
                "id": qid,
                "question": row["question"],
                "choice_labels": row["choices"]["label"],
                "choices": row["choices"]["text"],
                "correct_answer": row["answerKey"],
            }
            f.write(json.dumps(entry) + "\n")
            n_kept += 1

    print(f"\nWrote {n_kept} train examples to {output_path}")
    if test_qids:
        print(f"  Train ∩ Test = {n_dropped_overlap} qid(s); asserting disjointness (must be 0).")
        if n_dropped_overlap != 0:
            raise SystemExit(
                f"FAIL: ARC train ∩ test = {n_dropped_overlap} (expected 0). "
                "Aborting -- the eval would be contaminated."
            )
    else:
        print("  (Test split not yet present; skipping disjointness assertion.)")

    # Verify by reading back first 3 lines
    print("\nVerification (first 3 lines):")
    with open(output_path) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            obj = json.loads(line)
            print(f"  [{i}] id={obj['id']} question={obj['question'][:80]}...")
            print(f"       labels={obj['choice_labels']}, answer={obj['correct_answer']}")


if __name__ == "__main__":
    main()
