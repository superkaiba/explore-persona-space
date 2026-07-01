#!/usr/bin/env python
"""Issue #816 pod-side upload — Exp-4 adapters + Exp-5 tensors + raw generations.

Runs at the end of ``issue816_dispatch.sh`` (after Phase A, before pod release).
Uploads:
  - Exp-4 preventative LoRA adapters (canonical artifact) -> HF MODEL repo, one
    folder commit per cell under ``issue816_<slug>/adapters/{cell}/``.
  - Exp-5 per-dataset projection-difference PREDICTOR tensors (the OFF-POD null
    battery consumes them as downstream inputs) + the per-draw x per-axis null
    matrices, and any saved random directions -> HF DATA repo
    ``issue816_<slug>/analysis_tensors/`` (Upload Policy "Intermediate analysis
    tensors ... before pod termination" #521).
  - RAW generations: the Exp-2 steering + Exp-4 post-ft eval JSONs (each row
    carries the model's own generated ``response`` + per-draw diagnostics) ->
    HF DATA repo ``issue816_<slug>/raw_completions/`` via a single bulk
    ``upload_folder`` commit, so the model's own completions land off-pod before
    termination (Upload Policy raw-completions contract). The AGGREGATE eval JSONs
    also stay in git ``eval_results/issue_816/`` (JSON/text; verifier syncs Step 8).

Emits the ``reproducibility_card`` (per-cell adapter_paths verified on HF +
wandb_project/wandb_run_names) the sentinel needs (Step-7 contract). Fail-loud:
any upload that does not verify on a fresh Hub listing raises.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# load_dotenv() BEFORE any huggingface_hub import (#745 import-order): the Hub
# accelerator envs (HF_XET_HIGH_PERFORMANCE / HF_HUB_ENABLE_HF_TRANSFER) are
# frozen at huggingface_hub import time, so the env setup must precede it.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import issue778_lib as lib  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.hub import (  # noqa: E402
    DEFAULT_DATASET_REPO,
    DEFAULT_MODEL_REPO,
    list_repo_files_complete,
    upload_model,
)

WANDB_ENTITY = "superkaiba1"  # the account HF+WandB run under; read at run time when possible


def _verify_prefix(repo_id: str, repo_type: str, prefix: str, min_files: int = 1) -> int:
    files = list_repo_files_complete(HfApi(), repo_id, repo_type=repo_type, revision="main")
    hits = [f for f in files if f.startswith(prefix)]
    if len(hits) < min_files:
        raise RuntimeError(
            f"upload verify FAILED: expected >={min_files} files under "
            f"{repo_id}/{prefix}, found {len(hits)}"
        )
    return len(hits)


def _resolve_wandb_entity() -> str:
    """Read the live WandB default entity; fall back to the account constant."""
    try:
        import wandb

        ent = wandb.Api().default_entity
        if ent:
            return str(ent)
    except Exception:
        pass
    return WANDB_ENTITY


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #816 pod-side upload.")
    parser.add_argument("--issue", type=int, default=816)
    parser.add_argument("--slug", default="persona_vectors")
    parser.add_argument("--out-root", default="eval_results/issue_816")
    parser.add_argument("--ckpt-root", default="checkpoints/issue_816")
    parser.add_argument("--tensor-root", default="data/issue_816/store")
    args = parser.parse_args()

    exp_name = f"issue{args.issue}_{args.slug}"
    out_root = Path(args.out_root)
    ckpt_root = Path(args.ckpt_root)
    tensor_root = Path(args.tensor_root)

    summary: dict = {"adapters": {}, "analysis_tensors": {}, "raw_completions": {}}
    adapter_paths: dict[str, str] = {}
    wandb_run_names: list[str] = []

    # ── Exp-4 LoRA adapters -> HF MODEL repo (one folder commit per cell) ─────
    if ckpt_root.exists():
        for cell_dir in sorted(p for p in ckpt_root.iterdir() if p.is_dir()):
            cell = cell_dir.name
            # Skip a dir with no adapter weights (e.g. a failed/empty cell).
            if not any(cell_dir.glob("adapter_model.safetensors")):
                continue
            path_in_repo = f"{exp_name}/adapters/{cell}"
            url = upload_model(
                str(cell_dir),
                repo_id=DEFAULT_MODEL_REPO,
                path_in_repo=path_in_repo,
                ignore_patterns=[
                    "checkpoint-*",
                    "optimizer.pt",
                    "scheduler.pt",
                    "train_prepared.jsonl",
                ],
            )
            if not url:
                raise RuntimeError(f"adapter upload returned no path for {cell}")
            n = _verify_prefix(DEFAULT_MODEL_REPO, "model", path_in_repo, min_files=1)
            adapter_paths[cell] = path_in_repo
            wandb_run_names.append(f"issue816_{cell}")
            summary["adapters"][cell] = {"path_in_repo": path_in_repo, "n_files": n}
            print(f"[upload] adapter {cell} -> {path_in_repo} ({n} files)", flush=True)

    # ── Exp-5 analysis tensors + null matrices -> HF DATA repo ────────────────
    at_prefix = f"{exp_name}/analysis_tensors"
    n_tensors = 0
    if tensor_root.exists():
        for pt in sorted([*tensor_root.rglob("*.pt"), *tensor_root.rglob("*.json")]):
            rel = pt.relative_to(tensor_root)
            dest = f"{at_prefix}/{rel}"
            url = hub._upload(
                pt,
                repo_id=DEFAULT_DATASET_REPO,
                repo_type="dataset",
                path_in_repo=dest,
                upload_as_file=True,
            )
            if not url:
                raise RuntimeError(f"analysis-tensor upload returned no path for {pt}")
            n_tensors += 1
    if n_tensors:
        n = _verify_prefix(DEFAULT_DATASET_REPO, "dataset", at_prefix, min_files=1)
        summary["analysis_tensors"] = {"prefix": at_prefix, "n_files": n}
        print(f"[upload] analysis tensors -> {at_prefix} ({n} files)", flush=True)

    # ── RAW generations (steering + preventative eval JSONs) -> HF DATA repo ──
    # The model's own generated responses live in these per-cell eval JSONs; land
    # them off-pod before termination (raw-completions Upload Policy). One bulk
    # upload_folder commit per subdir (well under the 256-commits/hr cap).
    rc_prefix = f"{exp_name}/raw_completions"
    for sub in ("steering", "preventative"):
        sdir = out_root / sub
        if not sdir.exists() or not any(sdir.glob("*.json")):
            continue
        dest = f"{rc_prefix}/{sub}"
        url = hub._upload(
            sdir,
            repo_id=DEFAULT_DATASET_REPO,
            repo_type="dataset",
            path_in_repo=dest,
        )
        if not url:
            raise RuntimeError(f"raw-generation upload returned no path for {sub}")
        n = _verify_prefix(DEFAULT_DATASET_REPO, "dataset", dest, min_files=1)
        summary["raw_completions"][sub] = {"prefix": dest, "n_files": n}
        print(f"[upload] raw generations {sub} -> {dest} ({n} files)", flush=True)

    # ── reproducibility_card for the sentinel (Step-7 contract) ───────────────
    summary["reproducibility_card"] = {
        "adapter_paths": adapter_paths,
        "wandb_project": "issue816",
        "wandb_entity": _resolve_wandb_entity(),
        "wandb_run_names": sorted(wandb_run_names),
        "hf_model_repo": DEFAULT_MODEL_REPO,
        "hf_data_repo": DEFAULT_DATASET_REPO,
    }
    summary["reproducibility"] = lib.repro_metadata()
    summary["hf_model_repo"] = DEFAULT_MODEL_REPO
    summary["hf_data_repo"] = DEFAULT_DATASET_REPO
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
