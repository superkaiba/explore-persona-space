#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (×, →, —, ≥) in scientific docstrings + log messages.
"""Merge a (pair, seed) LoRA adapter into the base model and upload to HF Hub.

Per plan v3 §4.6. The outcome eval (``scripts/issue404_outcome_eval.py``)
expects merged checkpoints at::

    {ISSUE404_MODEL_REPO}/issue404_pair_{pair}_seed{seed}/

where ``ISSUE404_MODEL_REPO`` is ``superkaiba1/explore-persona-space``.
This script writes those subfolders. It is the missing post-training step
in the round-1 implementation: the outcome eval was attempting to load
``repo@revision`` strings directly with vLLM, which does not parse that
form. With this script in the pipeline the outcome eval can load each
cell via standard ``snapshot_download(allow_patterns=...)`` then
``vllm.LLM(model=<local_dir>)``.

Mirrors the project's existing merge+upload pattern (e.g.
``scripts/run_trait_transfer.py`` + ``scripts/run_100_persona_leakage.py``):
``train.sft.merge_lora`` to bake the adapter into the base, then
``orchestrate.hub.upload_model`` to push the merged dir to a per-cell
subfolder of the shared model repo.

Usage::

    # Merge + upload one cell:
    uv run python scripts/issue404_merge_and_upload.py \
        --pair insecure_code --seed 0 \
        --adapter-dir /workspace/sft_runs/issue404_insecure_code_seed0/coupling_adapter

    # Merge + upload many cells (one per line in a TSV):
    #   pair<TAB>seed<TAB>adapter_dir
    uv run python scripts/issue404_merge_and_upload.py --from-tsv cells.tsv

    # Dry-run (skip the HF Hub upload; useful for local smoke-testing):
    uv run python scripts/issue404_merge_and_upload.py --pair insecure_code \
        --seed 0 --adapter-dir ... --no-upload
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402
from issue404_common import (  # noqa: E402
    ISSUE404_MODEL_REPO,
    PAIRS,
    issue404_adapter_subfolder,
    reproducibility_metadata,
)

load_dotenv()

logger = logging.getLogger("issue404_merge_and_upload")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MERGED_CACHE_DIR = PROJECT_ROOT / "models" / "issue_404_merged"


def merge_and_upload_cell(
    pair: str,
    seed: int,
    adapter_dir: Path,
    repo_id: str,
    base_model: str,
    *,
    gpu_id: int,
    no_upload: bool,
    delete_local_after_upload: bool,
) -> dict:
    """Merge one (pair, seed) adapter into the base and push to HF Hub.

    Returns a dict carrying the local merged dir + remote subfolder + repo
    so the caller can persist a manifest. Raises ``RuntimeError`` loudly
    on any upload failure (CLAUDE.md fail-loud Upload Policy).
    """
    # Defer the heavy imports until the function actually runs — that way
    # the script's ``--help`` is fast and a TSV-driven multi-cell loop can
    # log each cell's start before paying the CUDA-init cost.
    from explore_persona_space.orchestrate.hub import upload_model
    from explore_persona_space.train.sft import merge_lora

    if pair not in PAIRS:
        raise ValueError(f"pair={pair!r} not in PAIRS={PAIRS}")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter dir does not exist: {adapter_dir}")
    if not any(adapter_dir.glob("adapter_config.json")):
        # PEFT adapters always emit an adapter_config.json next to the
        # safetensor weights; absence here means the caller pointed us at
        # the wrong directory (e.g. a checkpoint-NNN subdir or a merged
        # dir). Fail loud rather than upload something invalid.
        raise FileNotFoundError(
            f"No adapter_config.json under {adapter_dir}; this does not look "
            "like a PEFT LoRA adapter dir. Expected dir produced by train_lora."
        )

    subfolder = issue404_adapter_subfolder(pair, seed)
    merged_dir = MERGED_CACHE_DIR / subfolder
    merged_dir.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Cell pair=%s seed=%d: merging adapter %s into %s, output %s",
        pair,
        seed,
        adapter_dir,
        base_model,
        merged_dir,
    )
    merge_lora(
        base_model_path=base_model,
        adapter_path=str(adapter_dir),
        output_dir=str(merged_dir),
        gpu_id=gpu_id,
    )

    # Sanity-check the merge produced the expected Transformers shape.
    needed = ["config.json", "tokenizer_config.json"]
    missing = [n for n in needed if not (merged_dir / n).exists()]
    if missing:
        raise RuntimeError(
            f"Merge for pair={pair} seed={seed} did not produce expected "
            f"files under {merged_dir}; missing {missing}."
        )

    if no_upload:
        logger.info("--no-upload set; skipping HF Hub upload for pair=%s seed=%d", pair, seed)
        uploaded_to = ""
    else:
        logger.info(
            "Cell pair=%s seed=%d: uploading %s -> %s/%s",
            pair,
            seed,
            merged_dir,
            repo_id,
            subfolder,
        )
        uploaded_to = upload_model(
            model_path=str(merged_dir),
            repo_id=repo_id,
            path_in_repo=subfolder,
            delete_after=delete_local_after_upload,
        )
        if not uploaded_to:
            # upload_model returns "" on any failure (HF_TOKEN missing,
            # 4xx, verification mismatch). Per CLAUDE.md Upload Policy,
            # treat as fatal — never let a "no-op upload" silently
            # advance the pipeline.
            raise RuntimeError(
                f"upload_model returned empty for pair={pair} seed={seed}; "
                f"local merged dir preserved at {merged_dir}."
            )

    return {
        "pair": pair,
        "seed": seed,
        "adapter_dir": str(adapter_dir),
        "base_model": base_model,
        "merged_local_dir": str(merged_dir),
        "repo_id": repo_id,
        "subfolder": subfolder,
        "hf_upload_path": uploaded_to,
        "deleted_local_after_upload": bool(delete_local_after_upload and uploaded_to),
    }


def load_tsv(path: Path) -> list[tuple[str, int, Path]]:
    """Parse a TSV with columns ``pair<TAB>seed<TAB>adapter_dir`` per line.

    Skips blank lines + ``#`` comment lines. Adapter dirs are resolved as
    absolute paths.
    """
    cells: list[tuple[str, int, Path]] = []
    with open(path) as f:
        for n, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"{path}:{n}: expected 3 tab-separated fields "
                    f"(pair, seed, adapter_dir), got {len(parts)}: {s!r}"
                )
            pair_, seed_str, adapter_str = parts
            cells.append((pair_, int(seed_str), Path(adapter_str).resolve()))
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pair", choices=PAIRS, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Local PEFT LoRA adapter dir (the dir containing adapter_config.json).",
    )
    parser.add_argument(
        "--from-tsv",
        type=Path,
        default=None,
        help="TSV with columns: pair<TAB>seed<TAB>adapter_dir per line.",
    )
    parser.add_argument("--repo-id", default=ISSUE404_MODEL_REPO)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip the HF Hub upload (dry-run; keeps local merged dir).",
    )
    parser.add_argument(
        "--delete-local-after-upload",
        action="store_true",
        help=(
            "Delete the local merged dir after a verified upload. "
            "Reduces pod disk pressure when many cells are processed."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional path to write a JSONL manifest of "
            "{pair, seed, repo_id, subfolder, hf_upload_path, ...} rows."
        ),
    )
    parser.add_argument(
        "--cleanup-merged-cache",
        action="store_true",
        help=(
            "Delete the entire local merged-cache dir after a successful run. "
            "Use ONLY after every cell uploaded successfully."
        ),
    )
    args = parser.parse_args()

    # Resolve the cell list — either one (--pair/--seed/--adapter-dir) or
    # many (--from-tsv).
    cells: list[tuple[str, int, Path]] = []
    if args.from_tsv is not None:
        if args.pair is not None or args.seed is not None or args.adapter_dir is not None:
            raise ValueError("--from-tsv is mutually exclusive with --pair/--seed/--adapter-dir")
        cells = load_tsv(args.from_tsv)
    else:
        if args.pair is None or args.seed is None or args.adapter_dir is None:
            raise ValueError("Must specify either --from-tsv OR all of --pair/--seed/--adapter-dir")
        cells = [(args.pair, args.seed, args.adapter_dir.resolve())]

    if not cells:
        logger.warning("No cells to process; exiting.")
        return 0

    logger.info("Merging + uploading %d cell(s) to %s", len(cells), args.repo_id)
    manifest_rows: list[dict] = []
    for pair_, seed_, adapter_dir_ in cells:
        row = merge_and_upload_cell(
            pair=pair_,
            seed=seed_,
            adapter_dir=adapter_dir_,
            repo_id=args.repo_id,
            base_model=args.base_model,
            gpu_id=args.gpu_id,
            no_upload=args.no_upload,
            delete_local_after_upload=args.delete_local_after_upload,
        )
        row["metadata"] = reproducibility_metadata({"script": "issue404_merge_and_upload"})
        manifest_rows.append(row)

        # Persist the manifest incrementally — even a single cell, so a
        # crash after upload-N preserves upload-1..N-1's record.
        if args.manifest is not None:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            with open(args.manifest, "a") as mf:
                mf.write(json.dumps(row) + "\n")

    if args.cleanup_merged_cache and MERGED_CACHE_DIR.exists():
        # Sanity check: only nuke the cache if every cell uploaded.
        if any(r["hf_upload_path"] == "" for r in manifest_rows):
            logger.warning(
                "--cleanup-merged-cache set but at least one cell skipped upload; "
                "leaving %s intact.",
                MERGED_CACHE_DIR,
            )
        else:
            shutil.rmtree(MERGED_CACHE_DIR, ignore_errors=True)
            logger.info("Deleted local merged-cache dir %s", MERGED_CACHE_DIR)

    logger.info("Merge + upload done for %d cell(s).", len(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
