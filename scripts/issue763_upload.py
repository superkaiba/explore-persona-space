#!/usr/bin/env python3
"""Issue #763: upload raw completions + v0/r_B analysis tensors to HF (pre-teardown).

Per the Upload Policy (CLAUDE.md): raw completions AND plan-referenced analysis
tensors MUST land on the HF data repo BEFORE the GPU pod is released. It:

1. uploads every ``raw_completions.json`` under ``eval_results/issue_763/`` via
   the canonical bulk helper ``upload_raw_completions_to_data_repo`` (ONE
   ``upload_folder`` commit; #664 per-file-loop trap avoided);
2. bulk-uploads the v0 + r_B ``.pt`` shards to
   ``issue763_matched_v0/analysis_tensors/`` (the plan-named downstream inputs —
   losing them makes the analysis unrunnable, #521);
3. writes a SENTINEL that ``poll_pipeline.py`` drains (the only pod-side ->
   orchestrator channel; pod code NEVER shells scripts/task.py).

Two invocations in the dispatch (#763 CONCERN premature-results-sentinel):

- ``--progress-only`` — runs WHILE the GPU pod is live (raw completions +
  rollouts must land before teardown) and writes a NON-FINAL
  ``epm:upload-progress`` sentinel. An observing orchestrator must NOT see
  ``epm:results`` here, because judge/fit/figures have not produced their
  primary deliverables yet.
- (default, no flag) — runs LAST, AFTER fit + figures exist, re-uploads the
  full artifact set (now including the captured r_B), and writes the
  ``epm:results`` END-OF-RUN sentinel.

Usage::

    uv run python scripts/issue763_upload.py --progress-only   # pre-teardown
    uv run python scripts/issue763_upload.py                    # final, end-of-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue763_common import (  # noqa: E402
    DATA_DIR,
    EVAL_RESULTS_DIR,
    EXPERIMENT_NAME,
    HF_ANALYSIS_TENSORS_PREFIX,
    HF_DATA_REPO,
    HF_OVERFLOW_REPO,
    is_storage_quota_403,
    write_sentinel,
)

from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo  # noqa: E402

logger = logging.getLogger("issue763_upload")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# The analysis-input artifacts uploaded under ``issue763_matched_v0/
# analysis_tensors/<sub>/`` so the git-clone-only off-pod judge VM can
# ``snapshot_download`` them. Each entry = (sub-dir name, local source root,
# glob pattern). ``pv_rollouts`` (#763 BLOCKER pv-rollouts-not-uploaded) carries
# the per-rollout JSONLs the OFF-pod ``--phase judge`` reads (the GPU pod is
# stopped during the judge poll), so it MUST land BEFORE the pod stops — it is
# uploaded in the ``--progress-only`` pre-teardown pass like the v0 shards.
# ``pv_judge`` carries the off-pod judge's keep-flag JSONs so the RESUMED pod's
# ``--phase capture`` can ``snapshot_download`` them; it is produced off-pod, so
# it only exists at the final (no-flag) upload.
_ANALYSIS_ARTIFACTS: tuple[tuple[str, Path, str], ...] = (
    ("v0_shards", EVAL_RESULTS_DIR, "*.pt"),
    ("pv_shards", EVAL_RESULTS_DIR, "*.pt"),
    ("pv_rollouts", DATA_DIR, "*.jsonl"),
    # pv_judge/ = LEGACY r3 keep-flags (alignment-rubric contaminated); kept for
    # provenance. pv_judge_v2/ = the CANONICAL keep-flags (corrected trait rubric,
    # stamped with judge_system_prompt_hash) — it has a dedicated upload in
    # issue763_extract_pv_rb._upload_pv_judge_v2, but list it here too so this
    # general analysis-tensors pass also lands it (defense-in-depth; avoids a
    # canonical keep-flag set silently missing if the dedicated pass is skipped).
    ("pv_judge", DATA_DIR, "*.json"),
    ("pv_judge_v2", DATA_DIR, "*.json"),
    # gen/<behavior>/<context>.json — the E0 generated completions that the
    # phase-2 E0 judge consumes. The gate-split means the GPU pod that wrote
    # gen/ is deleted before phase 2 starts; without this entry the phase-2
    # E0 judge crashed with FileNotFoundError on a fresh VM (hotfix 2026-06-30
    # after the first phase-2 attempt died at [phase=judge]).
    ("gen", DATA_DIR, "*/*.json"),
)


def _upload_analysis_tensors() -> dict:
    """Bulk upload the analysis-input artifacts (ONE upload_folder commit each).

    Covers the v0 + r_B ``.pt`` shards AND the PV rollouts (``pv_rollouts/
    <behavior>.jsonl``) + the off-pod judge keep-flags (``pv_judge/
    <behavior>.json``) — #763 BLOCKER pv-rollouts-not-uploaded: the off-pod
    judge (run on the VM while the GPU pod is stopped) fetches the rollouts via
    ``snapshot_download`` of this issue-owned prefix, so they must be uploaded
    before the pod stops.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    uploaded = {}
    for sub, src_root, pattern in _ANALYSIS_ARTIFACTS:
        local = src_root / sub
        if not local.is_dir() or not any(local.glob(pattern)):
            continue
        path_in_repo = f"{HF_ANALYSIS_TENSORS_PREFIX}/{sub}"
        repo_used = HF_DATA_REPO
        try:
            api.upload_folder(
                folder_path=str(local),
                path_in_repo=path_in_repo,
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                allow_patterns=[pattern],
                commit_message=f"issue763: analysis tensors {sub}",
            )
        except Exception as e:
            if not is_storage_quota_403(e):
                raise
            logger.warning("HF storage 403 on %s; overflow repo", sub)
            repo_used = HF_OVERFLOW_REPO
            api.upload_folder(
                folder_path=str(local),
                path_in_repo=path_in_repo,
                repo_id=HF_OVERFLOW_REPO,
                repo_type="dataset",
                allow_patterns=[pattern],
                commit_message=f"issue763: analysis tensors {sub} (overflow)",
            )
        files = [
            f
            for f in api.list_repo_files(repo_used, repo_type="dataset")
            if f.startswith(path_in_repo)
        ]
        uploaded[sub] = {"repo": repo_used, "path_in_repo": path_in_repo, "n_files": len(files)}
        logger.info("uploaded %d %s files -> %s/%s", len(files), sub, repo_used, path_in_repo)

    # Top-level fit-input JSONs — E0 (the judged rates) + pv_rb (the PV summary).
    # These are the fit's y source + PV baseline; the --from-phase fit resume
    # stages them from this exact prefix on a fresh CPU VM
    # (issue763_fit_predictors._stage_fit_inputs_from_hf). They are non-LFS text
    # JSON, so they ride the quota-safe git-blob path (never LFS) — upload_file
    # per file (single-file API; NOT upload_folder, which no-ops on a file path).
    for name in ("E0_matched_by_behavior.json", "pv_rb_by_behavior.json"):
        local = EVAL_RESULTS_DIR / name
        if not local.is_file():
            continue
        dst = f"{HF_ANALYSIS_TENSORS_PREFIX}/{name}"
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=dst,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            commit_message=f"issue763: fit input {name}",
        )
        uploaded[name] = {"repo": HF_DATA_REPO, "path_in_repo": dst, "n_files": 1}
        logger.info("uploaded %s -> %s/%s", name, HF_DATA_REPO, dst)
    return uploaded


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #763: upload artifacts + sentinel.")
    ap.add_argument(
        "--progress-only",
        action="store_true",
        help="pre-teardown upload; write epm:upload-progress (NOT the final epm:results)",
    )
    ap.add_argument(
        "--emit-gate",
        metavar="GATE_NAME",
        default=None,
        help=(
            "write ONLY a blocking gate sentinel (no upload) so poll_pipeline.py "
            "surfaces status=gate; the orchestrator stops the pod, runs the "
            "off-pod judge on the VM, resumes, and re-dispatches at "
            "--from-phase pv_capture (#763 BLOCKER pv-judge-not-off-pod)"
        ),
    )
    args = ap.parse_args()

    # --emit-gate: pure signalling (no upload) — the rollouts already landed via
    # the preceding --progress-only pass; this just parks the orchestrator at the
    # off-pod-judge gate. blocks_pipeline=True ends the poll loop (Step 6d.4).
    if args.emit_gate:
        gate_note = {
            "task_id": 763,
            "experiment_name": EXPERIMENT_NAME,
            "gate": args.emit_gate,
            "phase": "pv phase-1 done; GPU pod must STOP for the off-pod PV judge",
            "resume_phase": "pv_capture",
        }
        path = write_sentinel(
            "epm:gate", gate_note, task_id=763, gate=args.emit_gate, blocks_pipeline=True
        )
        logger.info("wrote gate sentinel (gate=%s) -> %s", args.emit_gate, path)
        print(f"[issue763.upload] gate={args.emit_gate} sentinel={path}")
        return 0

    raw_map = upload_raw_completions_to_data_repo(
        experiment_name=EXPERIMENT_NAME,
        eval_results_dir=EVAL_RESULTS_DIR,
    )
    logger.info("uploaded %d raw_completions files", len(raw_map))

    tensors = _upload_analysis_tensors()

    note = {
        "task_id": 763,
        "experiment_name": EXPERIMENT_NAME,
        "n_raw_completions_files": len(raw_map),
        "analysis_tensors": tensors,
        "hf_data_repo": HF_DATA_REPO,
        "reproducibility_card": {
            "raw_completions_prefix": f"{EXPERIMENT_NAME}/raw_completions",
            "analysis_tensors_prefix": HF_ANALYSIS_TENSORS_PREFIX,
            "no_training": True,  # base-model read; no adapters / WandB runs
            "note": "base-model-only predictor re-measurement; no trained adapters",
        },
    }
    # CONCERN premature-results-sentinel: the pre-teardown upload writes a
    # NON-FINAL sentinel; the END-OF-RUN epm:results is written only by the
    # final (no-flag) invocation, after fit + figures have landed.
    if args.progress_only:
        kind = "epm:upload-progress"
        note["phase"] = "pre-teardown upload (raw completions + rollouts); deliverables pending"
    else:
        kind = "epm:results"
    path = write_sentinel(kind, note, task_id=763)
    logger.info("wrote %s sentinel -> %s", kind, path)
    print(f"[issue763.upload] {kind} raw={len(raw_map)} tensors={tensors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
