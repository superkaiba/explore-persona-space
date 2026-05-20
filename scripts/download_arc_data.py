#!/usr/bin/env python3
"""Download ARC-Challenge dataset and convert to JSONL format for logprob eval.

Outputs one JSON object per line with the schema expected by
`evaluate_capability_logprob()` in `src/explore_persona_space/eval/capability.py`:

    {"question": "...", "choice_labels": ["A", "B", "C", "D"],
     "choices": ["choice text 1", ...], "correct_answer": "B"}
"""

import json
from pathlib import Path

from datasets import load_dataset


def main():
    # Determine project root (this script lives in scripts/)
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "raw" / "arc_challenge"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test.jsonl"

    print("Downloading ARC-Challenge from allenai/ai2_arc ...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

    print(f"Downloaded {len(ds)} examples.")
    print(f"Columns: {ds.column_names}")
    print(f"\nFirst 3 examples (raw):")
    for i in range(min(3, len(ds))):
        print(f"  [{i}] {ds[i]}")

    # Convert to expected format
    count = 0
    with open(output_path, "w") as f:
        for row in ds:
            entry = {
                "question": row["question"],
                "choice_labels": row["choices"]["label"],
                "choices": row["choices"]["text"],
                "correct_answer": row["answerKey"],
            }
            f.write(json.dumps(entry) + "\n")
            count += 1

    print(f"\nWrote {count} examples to {output_path}")

    # Verify by reading back first 3 lines
    print("\nVerification (first 3 lines):")
    with open(output_path) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            obj = json.loads(line)
            print(f"  [{i}] question={obj['question'][:80]}...")
            print(f"       labels={obj['choice_labels']}, answer={obj['correct_answer']}")
            print(
                f"       choices={[c[:40] + '...' if len(c) > 40 else c for c in obj['choices']]}"
            )


if __name__ == "__main__":
    main()
