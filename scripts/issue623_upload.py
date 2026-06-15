#!/usr/bin/env python3
"""Issue #623 — upload pod artifacts to the HF data repo before pod termination.

Uploads the intermediate analysis tensors (persona centroids + sycophancy trait
vector + panel_prompts.json) and the steering-probe raw completions to the HF
data repo under ``issue623_persona_vectors/`` (per CLAUDE.md Upload Policy: these
.pt files are "intermediate analysis tensors plan-referenced as downstream
inputs" since the off-pod analysis depends on them).

Fail-loud: ``hub._upload`` returns "" on failure; this script asserts a non-empty
returned path for every uploaded folder/file and exits non-zero otherwise, so the
dispatcher's ``set -e`` aborts before ``[phase=done]``.

The eval_results/issue_623/*.json + figures go to GIT on the issue branch
(committed VM-side), NOT here.

Usage (pod, normal exit path AFTER phases 1->5, BEFORE [phase=done]):
  uv run python scripts/issue623_upload.py \
      --persona-vectors-dir data/persona_vectors/issue623
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Hoist the hub import to module top (gotchas.md "lazy imports in smoke-skipped
# branches"): if this symbol drifts, the script fails at process start, not after
# the GPU phases.
from explore_persona_space.experiments.persona_decomp_623 import (  # noqa: E402
    HF_UPLOAD_PREFIX,
    repo_root_from_module,
)
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402
from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO, _upload  # noqa: E402


def upload_folder(local: Path, path_in_repo: str) -> str:
    """Upload a folder to the data repo; fail-loud on empty return."""
    if not local.exists():
        raise FileNotFoundError(f"Upload source missing: {local}")
    dest = _upload(
        local_path=local,
        repo_id=DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
    )
    if not dest:
        raise RuntimeError(f"Upload FAILED (empty return): {local} -> {path_in_repo}")
    print(f"[phase=upload] {local} -> {dest}", flush=True)
    return dest


def upload_file(local: Path, path_in_repo: str) -> str:
    """Upload a single file to the data repo; fail-loud on empty return."""
    if not local.exists():
        raise FileNotFoundError(f"Upload source missing: {local}")
    dest = _upload(
        local_path=local,
        repo_id=DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        upload_as_file=True,
    )
    if not dest:
        raise RuntimeError(f"Upload FAILED (empty return): {local} -> {path_in_repo}")
    print(f"[phase=upload] {local} -> {dest}", flush=True)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #623 — upload pod artifacts to HF.")
    parser.add_argument(
        "--persona-vectors-dir",
        default="data/persona_vectors/issue623",
        help="Dir with panel_prompts.json + method_a/ method_b/ sycophancy_trait/ "
        "steering_probe/ (relative to repo root).",
    )
    parser.add_argument(
        "--hf-prefix",
        default=HF_UPLOAD_PREFIX,
        help="HF data-repo path prefix for the upload (default the production "
        f"{HF_UPLOAD_PREFIX!r}; the dispatcher's --smoke-upload-only passes a "
        "_smoke/<ts> prefix so a smoke round-trip never pollutes the production tree).",
    )
    args = parser.parse_args()

    load_dotenv()
    repo_root = repo_root_from_module()
    base = (
        repo_root / args.persona_vectors_dir
        if not Path(args.persona_vectors_dir).is_absolute()
        else Path(args.persona_vectors_dir)
    )

    prefix = args.hf_prefix

    # panel_prompts.json (single file)
    upload_file(base / "panel_prompts.json", f"{prefix}/panel_prompts.json")

    # persona centroids: method_a (always), method_b (if produced)
    upload_folder(base / "method_a", f"{prefix}/persona_vectors/method_a")
    if (base / "method_b").exists():
        upload_folder(base / "method_b", f"{prefix}/persona_vectors/method_b")

    # sycophancy trait vectors + artifacts
    upload_folder(base / "sycophancy_trait", f"{prefix}/sycophancy_trait")

    # steering-probe raw completions (if produced)
    steering_raw = base / "steering_probe" / "raw_completions"
    if steering_raw.exists():
        upload_folder(steering_raw, f"{prefix}/steering_probe/raw_completions")

    print("[phase=upload] all uploads verified", flush=True)


if __name__ == "__main__":
    main()
