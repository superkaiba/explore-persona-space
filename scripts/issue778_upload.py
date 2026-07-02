#!/usr/bin/env python
"""Issue #778 pod-side upload — LoRA adapters + analysis tensors + eval JSONs.

Runs at the end of ``issue778_dispatch.sh`` (step 5). Uploads:
  - LoRA adapters (canonical artifact) -> HF MODEL repo, one folder commit per cell
    under ``issue778_<slug>/adapters/{cell}/``.
  - Intermediate analysis tensors the OFF-POD null battery consumes as downstream
    inputs (r_B, extraction pools, monitoring raw acts, finetune activations)
    -> HF DATA repo ``issue778_<slug>/analysis_tensors/`` (plan v2 §5 / Upload
    Policy "Intermediate analysis tensors ... before pod termination" #521).
  - eval JSONs (monitoring JSONL + finetune per-cell + meta) stay in git on the
    issue branch (JSON/text only) — the upload-verifier syncs them at Step 8; this
    script does NOT touch git.

Fail-loud: any upload that does not verify on a fresh Hub listing raises. Prints a
JSON summary the dispatcher threads into the sentinel's reproducibility_card.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv

# huggingface_hub.constants freezes HF_HUB_ENABLE_HF_TRANSFER at import time, so
# the env must be loaded BEFORE the hub import (#745 import-order lint).
load_dotenv()

import issue778_lib as lib  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

from explore_persona_space.orchestrate.hub import (  # noqa: E402
    DEFAULT_DATASET_REPO,
    DEFAULT_MODEL_REPO,
    list_repo_files_complete,
    upload_model,
)


def _verify_prefix(repo_id: str, repo_type: str, prefix: str, min_files: int = 1) -> int:
    # list_repo_files_complete's first positional is an HfApi instance (already
    # token-scoped); the repo_id is the SECOND positional. Passing repo_id first
    # bound it to the api slot and raised "missing 1 required positional
    # argument: 'repo_id'" (issue #778 epm:failure v3).
    files = list_repo_files_complete(HfApi(), repo_id, repo_type=repo_type, revision="main")
    hits = [f for f in files if f.startswith(prefix)]
    if len(hits) < min_files:
        raise RuntimeError(
            f"upload verify FAILED: expected >={min_files} files under "
            f"{repo_id}/{prefix}, found {len(hits)}"
        )
    return len(hits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #778 pod-side upload.")
    parser.add_argument("--issue", type=int, default=778)
    parser.add_argument("--slug", default="persona_vectors")
    parser.add_argument("--out-root", default="data/issue_778")
    parser.add_argument("--ckpt-root", default="checkpoints/issue_778")
    args = parser.parse_args()

    exp_name = f"issue{args.issue}_{args.slug}"
    out_root = Path(args.out_root)
    ckpt_root = Path(args.ckpt_root)

    summary: dict = {"adapters": {}, "analysis_tensors": {}}

    # ── LoRA adapters -> HF MODEL repo (one folder commit per cell) ──────────
    adapter_paths: dict[str, str] = {}
    if ckpt_root.exists():
        for cell_dir in sorted(p for p in ckpt_root.iterdir() if p.is_dir()):
            cell = cell_dir.name
            path_in_repo = f"{exp_name}/adapters/{cell}"
            url = upload_model(
                str(cell_dir),
                repo_id=DEFAULT_MODEL_REPO,
                path_in_repo=path_in_repo,
                ignore_patterns=["checkpoint-*", "optimizer.pt", "scheduler.pt"],
            )
            if not url:
                raise RuntimeError(f"adapter upload returned no path for {cell}")
            n = _verify_prefix(DEFAULT_MODEL_REPO, "model", path_in_repo, min_files=1)
            adapter_paths[cell] = path_in_repo
            summary["adapters"][cell] = {"path_in_repo": path_in_repo, "n_files": n}
            print(f"[upload] adapter {cell} -> {path_in_repo} ({n} files)", flush=True)

    # ── analysis tensors -> HF DATA repo analysis_tensors/ ───────────────────
    at_prefix = f"{exp_name}/analysis_tensors"
    tensor_dirs = [
        out_root / "rb",
        out_root / "activations",
        out_root / "monitoring",
        out_root / "finetune_activations",
    ]
    from explore_persona_space.orchestrate import hub

    n_uploaded = 0
    for tdir in tensor_dirs:
        if not tdir.exists():
            continue
        # Upload only .pt tensors (skip judge_cache / raw json under monitoring/).
        for pt in sorted(tdir.rglob("*.pt")):
            rel = pt.relative_to(out_root)
            dest = f"{at_prefix}/{rel}"
            # Single-file upload MUST pass upload_as_file=True (the folder branch
            # silently no-ops on a file path; hub._upload raises otherwise).
            url = hub._upload(
                pt,
                repo_id=DEFAULT_DATASET_REPO,
                repo_type="dataset",
                path_in_repo=dest,
                upload_as_file=True,
            )
            if not url:
                raise RuntimeError(f"analysis-tensor upload returned no path for {pt}")
            n_uploaded += 1
    if n_uploaded:
        n = _verify_prefix(DEFAULT_DATASET_REPO, "dataset", at_prefix, min_files=1)
        summary["analysis_tensors"] = {"prefix": at_prefix, "n_files": n}
        print(f"[upload] analysis tensors -> {at_prefix} ({n} files)", flush=True)

    summary["reproducibility"] = lib.repro_metadata()
    summary["hf_model_repo"] = DEFAULT_MODEL_REPO
    summary["hf_data_repo"] = DEFAULT_DATASET_REPO
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
