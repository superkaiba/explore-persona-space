#!/usr/bin/env python
"""Issue #778 amendment upload — corrected-monitoring-8prompt-ladder round.

Two phases (plan v3 §9):
  - ``pod``: runs at the end of the pod-side dispatch (after Leg A + Leg B, before
    the pod releases). Uploads the RAW last-prompt activation tensors the OFF-POD
    null battery re-projects (``monitoring_corrected/{trait}_acts.pt`` +
    ``monitoring_manyshot/{trait}_acts.pt``) to the HF DATA repo
    ``issue778_persona_vectors/analysis_tensors/`` AND the regenerated Leg-B
    exemplar pools (``exemplar_pool/{trait}_kept_pos.json``) to
    ``issue778_persona_vectors/extraction_rollouts_regen/`` (Upload Policy: plan-
    referenced downstream inputs before pod termination, #521). The eval JSONLs
    (``monitoring_corrected_*.jsonl`` / ``monitoring_manyshot_*.jsonl``) stay in
    git on the issue branch — this script does NOT touch git.
  - ``offpod``: runs on the VM after the null battery produces the per-draw x
    per-layer |r| matrices (``{trait}_{input_tag}_{corr}_{null}_draws.npy``);
    uploads them to ``analysis_tensors/null_draws/`` (the analyzer's honest-band
    recompute inputs, #521).

Fail-loud: any upload that does not verify on a fresh Hub listing raises. Prints a
JSON summary the dispatcher threads into the sentinel's reproducibility_card.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue778_lib as lib
from huggingface_hub import HfApi

from explore_persona_space.orchestrate import hub
from explore_persona_space.orchestrate.env import load_dotenv
from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO, list_repo_files_complete

load_dotenv()


def _verify_prefix(repo_id: str, repo_type: str, prefix: str, min_files: int = 1) -> int:
    files = list_repo_files_complete(HfApi(), repo_id, repo_type=repo_type, revision="main")
    hits = [f for f in files if f.startswith(prefix)]
    if len(hits) < min_files:
        raise RuntimeError(
            f"upload verify FAILED: expected >={min_files} files under "
            f"{repo_id}/{prefix}, found {len(hits)}"
        )
    return len(hits)


def _upload_file(local: Path, dest: str) -> None:
    url = hub._upload(
        local,
        repo_id=DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=dest,
        upload_as_file=True,
    )
    if not url:
        raise RuntimeError(f"upload returned no path for {local} -> {dest}")


def upload_pod_phase(out_root: Path, exp_name: str) -> dict:
    """Upload raw acts tensors + regenerated exemplar pools (pod-side, pre-teardown)."""
    summary: dict = {"analysis_tensors": {}, "exemplar_pools": {}}

    # Raw last-prompt activation tensors -> analysis_tensors/ (null re-projection).
    at_prefix = f"{exp_name}/analysis_tensors"
    n_at = 0
    for tag in ("monitoring_corrected", "monitoring_manyshot"):
        tdir = out_root / tag
        if not tdir.exists():
            continue
        for pt in sorted(tdir.rglob("*.pt")):
            rel = pt.relative_to(out_root)
            _upload_file(pt, f"{at_prefix}/{rel}")
            n_at += 1
    if n_at:
        n = _verify_prefix(DEFAULT_DATASET_REPO, "dataset", at_prefix, min_files=1)
        summary["analysis_tensors"] = {"prefix": at_prefix, "n_files_total": n, "n_uploaded": n_at}
        print(f"[upload] raw acts tensors -> {at_prefix} ({n_at} new)", flush=True)

    # Regenerated Leg-B exemplar pools -> extraction_rollouts_regen/.
    pool_prefix = f"{exp_name}/extraction_rollouts_regen"
    pool_dir = out_root / "exemplar_pool"
    n_pool = 0
    if pool_dir.exists():
        for pj in sorted(pool_dir.glob("*_kept_pos.json")):
            _upload_file(pj, f"{pool_prefix}/{pj.name}")
            n_pool += 1
    if n_pool:
        n = _verify_prefix(DEFAULT_DATASET_REPO, "dataset", pool_prefix, min_files=1)
        summary["exemplar_pools"] = {"prefix": pool_prefix, "n_files": n, "n_uploaded": n_pool}
        print(f"[upload] exemplar pools -> {pool_prefix} ({n_pool})", flush=True)

    return summary


def upload_offpod_phase(eval_root: Path, exp_name: str) -> dict:
    """Upload the null-draw |r| matrices (VM-side, after the null battery)."""
    summary: dict = {"null_draws": {}}
    nd_prefix = f"{exp_name}/analysis_tensors/null_draws"
    n_nd = 0
    for npy in sorted(eval_root.glob("*_draws.npy")):
        _upload_file(npy, f"{nd_prefix}/{npy.name}")
        n_nd += 1
    if n_nd:
        n = _verify_prefix(DEFAULT_DATASET_REPO, "dataset", nd_prefix, min_files=1)
        summary["null_draws"] = {"prefix": nd_prefix, "n_files": n, "n_uploaded": n_nd}
        print(f"[upload] null-draw matrices -> {nd_prefix} ({n_nd})", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #778 corrected-monitoring upload.")
    parser.add_argument("--issue", type=int, default=778)
    parser.add_argument("--slug", default="persona_vectors")
    parser.add_argument("--out-root", default="data/issue_778")
    parser.add_argument("--eval-results-root", default="eval_results/issue_778")
    parser.add_argument("--phase", choices=["pod", "offpod"], default="pod")
    args = parser.parse_args()

    exp_name = f"issue{args.issue}_{args.slug}"
    out_root = Path(args.out_root)
    eval_root = Path(args.eval_results_root)

    if args.phase == "pod":
        summary = upload_pod_phase(out_root, exp_name)
    else:
        summary = upload_offpod_phase(eval_root, exp_name)

    summary["phase"] = args.phase
    summary["hf_data_repo"] = DEFAULT_DATASET_REPO
    summary["reproducibility"] = lib.repro_metadata()
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
