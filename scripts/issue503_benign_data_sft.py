#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (×, →, —) in scientific docstrings + logs.
"""Issue #503 — Bucket D benign-data SFT launcher (plan v2 §4.5).

Per plan §4.5 "SFT recipe per selector × seed":
    lr=5e-5, batch=20, 5 epochs, AdamW, bf16. 3 random seeds {0, 42, 137}.
    LoRA r=32, α=256 (matching the rest of the spec for cross-bucket
    comparability — He et al. used full FT but LoRA suffices on Qwen-7B
    at this scale; verify in smoke).

This launcher reads each selector's top-100 JSONL (from
``issue503_benign_data_select.py``), materializes the SFT dataset for
that selector × seed, and prints the training command. Per the
``halt-criterion contract`` and ``no auto-launch from a CLI script`` we
print the command rather than spawn it; the orchestrator + experimenter
agent dispatches the run on a pod via the canonical ``scripts/train.py``
+ Hydra recipe path.

Inputs:

  --selector-jsonl <PATH>   # the per-selector top-K from issue503_benign_data_select.py
  --benign-corpus <PATH>    # the safety-filtered Alpaca/Dolly/GSM8K JSONL
  --seed <N>
  --selector-id <ID>

Outputs:

  data/issue503/benign_data/{selector_id}_seed{seed}.jsonl
    -- SFT-ready (prompt, completion) JSONL for the LoRA trainer
  + stdout: the canonical train command

WandB logging is required per CLAUDE.md code-style rule; pass
``report_to=wandb`` (the default in our TrainLoraConfig wraps this).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("issue503.benign_data_sft")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def output_dir(repo_root: Path) -> Path:
    p = repo_root / "data" / "issue503" / "benign_data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_selected_ids(selector_jsonl: Path) -> list[str]:
    ids: list[str] = []
    with selector_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids.append(str(obj["datapoint_id"]))
    if not ids:
        raise RuntimeError(f"Selector JSONL {selector_jsonl} contained no rows.")
    return ids


def load_corpus_by_id(corpus_jsonl: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with corpus_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows[str(obj["id"])] = obj
    return rows


def materialize_sft_dataset(
    selected_ids: list[str],
    corpus: dict[str, dict],
    *,
    out_path: Path,
) -> int:
    """Write a SFT-ready JSONL: each row {prompt, completion, metadata}."""
    n_written = 0
    with out_path.open("w") as fout:
        for dp_id in selected_ids:
            row = corpus.get(dp_id)
            if row is None:
                raise RuntimeError(
                    f"Selected id {dp_id!r} not in corpus; possibly stale selector output."
                )
            sft_row = {
                "id": dp_id,
                "prompt": row.get("instruction", ""),
                "completion": row.get("output", ""),
                "source": row.get("source", "alpaca"),
            }
            fout.write(json.dumps(sft_row) + "\n")
            n_written += 1
    return n_written


def build_train_command(
    selector_id: str,
    seed: int,
    sft_dataset_path: Path,
    out_adapter_subfolder: str,
) -> list[str]:
    """The canonical training command per plan §4.5.

    He et al. recipe: lr=5e-5, batch=20 (per-device 1 × grad_accum 20),
    5 epochs, AdamW, bf16, LoRA r=32 α=256.
    """
    return [
        "uv",
        "run",
        "python",
        "scripts/train.py",
        # The Hydra condition file for Bucket D selectors would live at
        # configs/condition/issue503_benign_data_sft.yaml; pass the per-cell
        # overrides inline below.
        "condition=issue503_benign_data_sft",
        f"seed={seed}",
        "training.learning_rate=5e-5",
        "training.per_device_train_batch_size=1",
        "training.gradient_accumulation_steps=20",
        "training.num_train_epochs=5",
        "training.bf16=true",
        "lora.r=32",
        "lora.lora_alpha=256",
        f"data.training_jsonl={sft_dataset_path}",
        f"experiment.upload_subfolder={out_adapter_subfolder}",
        f"experiment.tags=[issue503,bucket_D,{selector_id},seed{seed}]",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-jsonl", type=Path, required=True)
    parser.add_argument("--benign-corpus", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--selector-id",
        required=True,
        choices=["D0_random", "D1_representation", "D2_gradient", "D3_cosine", "D4_format"],
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the train command without running anything.",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    selected_ids = load_selected_ids(args.selector_jsonl)
    corpus = load_corpus_by_id(args.benign_corpus)
    logger.info(
        "Materializing SFT dataset: selector=%s seed=%d ids=%d corpus=%d",
        args.selector_id,
        args.seed,
        len(selected_ids),
        len(corpus),
    )

    out_path = output_dir(root) / f"{args.selector_id}_seed{args.seed}.jsonl"
    n_written = materialize_sft_dataset(selected_ids, corpus, out_path=out_path)
    logger.info("Wrote %d SFT rows to %s", n_written, out_path)

    out_subfolder = f"issue503_bucket_d_{args.selector_id}_seed{args.seed}/adapter"
    cmd = build_train_command(
        args.selector_id, args.seed, out_path, out_adapter_subfolder=out_subfolder
    )
    print(" ".join(cmd))
    return 0


if __name__ == "__main__":
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        logger.warning(
            "HF_TOKEN not set; the train command will fail when it tries to upload "
            "the resulting LoRA adapter to HF Hub. Make sure .env loads on the pod."
        )
    sys.exit(main())
