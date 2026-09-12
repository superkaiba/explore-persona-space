"""Freeze outcome-blind context selection from audited source manifests.

This preparation command reads source prompts and hashes only, never activation
targets, model predictions or task state. The output records selected IDs and
content hashes without copying prompt text. Run before experimental outcome reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from omegaconf import OmegaConf  # noqa: E402


def freeze_selection(audit_path: Path, config_path: Path) -> dict:
    """Validate source bytes, deduplicate by NFC content, and freeze selected IDs."""
    audit = json.loads(audit_path.read_text())
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if config["sampling"]["context_key"] != "sha256_of_utf8_NFC_prompt_without_whitespace_changes":
        raise ValueError("Unsupported context-key definition")
    if config["sampling"]["split_priority"] != ["train", "validation", "test"]:
        raise ValueError("Unsupported split ownership priority")
    if config["sampling"]["pilot_test"] != "always_disjoint_from_main_test":
        raise ValueError("Pilot test must be disjoint from main test")
    source_names = {
        "train": "train_25k.jsonl",
        "validation": "val_400.jsonl",
        "test": "test_1000.jsonl",
    }
    seen = {}
    eligible = {}
    exclusions = []
    source_records = {}
    seed = int(config["seed"])
    for split, filename in source_names.items():
        source = audit["manifest_sources"]["splits"][filename]
        data = Path(source["path"]).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha != source["sha256"]:
            raise ValueError(f"Source hash mismatch: {filename}")
        # JSONL is LF-delimited. str.splitlines would split embedded U+2028.
        rows = [json.loads(line) for line in data.decode("utf-8").split("\n") if line.strip()]
        if split == "train":
            rows = rows[:10000]
        selected = []
        for index, row in enumerate(rows):
            prompt = row["prompt"]
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"Empty or nontext prompt: {filename} row {index}")
            key = hashlib.sha256(unicodedata.normalize("NFC", prompt).encode("utf-8")).hexdigest()
            item = {
                "source_split": split,
                "source_row_index": index,
                "ladder_local_id": row["ladder_local_id"],
                "prompt_sha256": key,
            }
            if key in seen:
                exclusions.append({**item, "reason": "duplicate_content", "owner": seen[key]})
                continue
            seen[key] = {"split": split, "source_row_index": index}
            selected.append(item)
        selected.sort(
            key=lambda item: hashlib.sha256(f"{seed}:{item['prompt_sha256']}".encode()).hexdigest()
        )
        eligible[split] = selected
        source_records[split] = {
            "filename": filename,
            "sha256": sha,
            "n_source_rows": len(rows),
            "n_eligible": len(selected),
        }
    n_cal = int(config["sampling"]["calibration_prompts"])
    if len(eligible["train"]) < n_cal:
        raise ValueError("Insufficient training contexts for calibration")
    calibration = eligible["train"][:n_cal]
    eligible["train"] = eligible["train"][n_cal:]
    subsets = {"calibration": calibration}
    for split in source_names:
        n_pilot = int(config["sampling"]["pilot"][split])
        if len(eligible[split]) < n_pilot:
            raise ValueError(f"Insufficient eligible {split} contexts for pilot")
        subsets[f"pilot_{split}"] = eligible[split][:n_pilot]
        # All pilot rows are separate: stronger than only quarantining pilot test.
        remainder = eligible[split][n_pilot:]
        n_main = min(len(remainder), int(config["sampling"]["main_maximum"][split]))
        if n_main < 2:
            raise ValueError(f"Insufficient eligible {split} contexts for main")
        subsets[f"main_{split}"] = remainder[:n_main]
    all_keys = [row["prompt_sha256"] for subset in subsets.values() for row in subset]
    if len(all_keys) != len(set(all_keys)):
        raise AssertionError("Selected context content is not disjoint")
    return {
        "schema_version": "workspace-jr-selection-v1",
        "seed": seed,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "context_key_definition": "sha256(utf8(NFC(prompt))), no whitespace changes",
        "ordering_definition": "sha256(utf8(str(seed)+':'+context_key)) ascending",
        "source_records": source_records,
        "subsets": subsets,
        "counts": {name: len(rows) for name, rows in subsets.items()},
        "exclusions": exclusions,
        "uses_old_capture_survivors": False,
        "reads_experimental_outcomes": False,
    }


def main() -> None:
    """Write a new immutable selection manifest; refuse overwriting prior selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = freeze_selection(args.audit_file, args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "counts": record["counts"],
                "excluded_duplicate_rows": len(record["exclusions"]),
                "output": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
