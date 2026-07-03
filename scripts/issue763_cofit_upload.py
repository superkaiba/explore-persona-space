#!/usr/bin/env python3
"""Issue #763 `neutral-contrast-and-cofit`: HF uploads + pod-side sentinels.

Upload Policy (CLAUDE.md): the neutral-arm rollout TEXT + every plan-referenced
analysis tensor MUST land on the HF data repo BEFORE the GPU pod is released;
``discarded_artifacts: []`` (plan §10) — everything this round produces uploads.
One ``upload_folder`` commit per artifact dir (the #664 per-file-loop trap +
the 256-commits/hr cap both avoided). Fail-loud: a failed upload crashes the
phase (the upload-verifier is the safety net, never the only line of defense).

Three invocations (mirroring the parent ``issue763_upload.py`` pattern):

- ``--progress-only`` (Phase A tail, GPU pod still live): neutral rollouts
  (analysis_tensors/neutral_rollouts + raw_completions/neutral_arm), the
  per-rollout neutral means, the stripped-read arm means, the c0 shards, and
  the capture manifest. NON-final ``epm:upload-progress`` sentinel.
- ``--directions-only`` (Phase B tail, VM): neutral_judge keep-flags +
  pv_directions_v2 + neutral_arm_manifest (so Phase C can stage them anywhere).
- default (Phase C tail): the co-fit deliverables (cofit_results.json,
  nonlinear_tests.json, inputs_manifest.json, cofit_null_matrices) + a re-run
  of both passes above (idempotent), then the END-OF-RUN ``epm:results``
  sentinel naming the §6.5 primary deliverables.

``--emit-gate <name>`` writes a BLOCKING gate sentinel (the Phase-A pod-cycle
gate — the orchestrator stops the pod, runs the off-pod Phase-B judge on the
VM, then dispatches Phase C).

Usage::

    uv run python scripts/issue763_cofit_upload.py --progress-only
    uv run python scripts/issue763_cofit_upload.py --emit-gate cofit_phaseA_done
    uv run python scripts/issue763_cofit_upload.py --directions-only
    uv run python scripts/issue763_cofit_upload.py            # final, end-of-run
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
    C0_SHARD_DIR,
    COFIT_DIR,
    COFIT_NULL_MATRIX_DIR,
    HF_ANALYSIS_TENSORS_PREFIX,
    HF_DATA_REPO,
    HF_RAW_COMPLETIONS_PREFIX,
    NEUTRAL_JUDGE_DIR,
    NEUTRAL_ROLLOUT_DIR,
    NEUTRAL_ROLLOUT_MEANS_DIR,
    PV_DIRECTIONS_V2_DIR,
    smoke_scope_active,
    write_sentinel,
)

logger = logging.getLogger("issue763_cofit_upload")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# (sub-path-in-repo under analysis_tensors/, local dir, glob) — one commit each.
_PROGRESS_ARTIFACTS: tuple[tuple[str, Path, str], ...] = (
    ("neutral_rollouts", NEUTRAL_ROLLOUT_DIR, "*.jsonl"),
    ("neutral_rollout_means", NEUTRAL_ROLLOUT_MEANS_DIR, "*.pt"),
    ("arm_means", COFIT_DIR / "arm_means", "*.pt"),
    ("c0_shards", C0_SHARD_DIR, "*.pt"),
    # The capture manifest carries the LOAD-BEARING parity record Phase B's
    # assemble refuses to run without; the Phase-A instance is DELETED at the
    # gate, so it MUST ride the progress pass (review r1 C1(i)).
    ("cofit_manifests", COFIT_DIR, "capture_arm_means_manifest.json"),
)
_DIRECTIONS_ARTIFACTS: tuple[tuple[str, Path, str], ...] = (
    ("neutral_judge", NEUTRAL_JUDGE_DIR, "*.json"),
    ("pv_directions_v2", PV_DIRECTIONS_V2_DIR, "*.pt"),
    # Plan-required §6.5 deliverable: Phase C stages it for the cos/yield
    # panels + the final upload existence check (review r1 C2 / Codex C2).
    ("cofit_manifests", COFIT_DIR, "neutral_arm_manifest.json"),
)
_FINAL_ARTIFACTS: tuple[tuple[str, Path, str], ...] = (
    ("cofit_null_matrices", COFIT_NULL_MATRIX_DIR, "*.pt"),
    ("cofit_results", COFIT_DIR, "*.json"),
)


def _upload_dirs(artifacts: tuple[tuple[str, Path, str], ...]) -> list[str]:
    """One bulk ``upload_folder`` commit per non-empty artifact dir; fail-loud."""
    from huggingface_hub import HfApi

    api = HfApi()
    uploaded: list[str] = []
    for sub, local_dir, pattern in artifacts:
        if not local_dir.is_dir() or not any(local_dir.glob(pattern)):
            logger.info("[upload] %s: nothing at %s/%s (skip)", sub, local_dir, pattern)
            continue
        path_in_repo = f"{HF_ANALYSIS_TENSORS_PREFIX}/{sub}"
        api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=path_in_repo,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            allow_patterns=[pattern],
            commit_message=f"issue763 neutral-contrast-and-cofit: {sub}",
        )
        uploaded.append(path_in_repo)
        logger.info("[upload] %s -> %s/%s", sub, HF_DATA_REPO, path_in_repo)
    return uploaded


def _upload_neutral_raw_completions() -> str | None:
    """Neutral rollout TEXT -> raw_completions/neutral_arm/ (persist-by-default).

    The generation-and-reduce rule: the neutral generation's rollout text
    persists under the raw-completions bucket (non-LFS JSONL) IN ADDITION to the
    analysis_tensors mirror the Phase-B judge stages from.
    """
    from huggingface_hub import HfApi

    if not NEUTRAL_ROLLOUT_DIR.is_dir() or not any(NEUTRAL_ROLLOUT_DIR.glob("*.jsonl")):
        return None
    api = HfApi()
    path_in_repo = f"{HF_RAW_COMPLETIONS_PREFIX}/neutral_arm"
    api.upload_folder(
        folder_path=str(NEUTRAL_ROLLOUT_DIR),
        path_in_repo=path_in_repo,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=["*.jsonl"],
        commit_message="issue763 neutral-contrast-and-cofit: neutral-arm raw completions",
    )
    logger.info("[upload] raw completions -> %s/%s", HF_DATA_REPO, path_in_repo)
    return path_in_repo


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #763 neutral-contrast-and-cofit uploads.")
    ap.add_argument("--progress-only", action="store_true")
    ap.add_argument("--directions-only", action="store_true")
    ap.add_argument("--emit-gate", default=None)
    args = ap.parse_args()
    if smoke_scope_active():
        # Production-only script: under the smoke scope every path constant
        # points at smoke_scope/ residue — uploading THAT to the production HF
        # prefixes would be silent provenance corruption.
        raise RuntimeError(
            "issue763_cofit_upload must never run under EPM_ISSUE763_SMOKE_SCOPE=1 — "
            "the smoke is upload-free by design (dispatcher logs uploads as LOG-ONLY)"
        )

    if args.emit_gate:
        path = write_sentinel(
            "epm:progress",
            {
                "round": "neutral-contrast-and-cofit",
                "note": f"Phase A complete; parked at gate={args.emit_gate} "
                "(off-pod neutral judge next)",
            },
            gate=args.emit_gate,
            blocks_pipeline=True,
        )
        print(f"[issue763.cofit_upload] gate sentinel -> {path}")
        return 0

    if args.progress_only:
        uploaded = _upload_dirs(_PROGRESS_ARTIFACTS)
        raw = _upload_neutral_raw_completions()
        write_sentinel(
            "epm:upload-progress",
            {
                "round": "neutral-contrast-and-cofit",
                "uploaded": uploaded + ([raw] if raw else []),
                "final": False,
            },
        )
        print(f"[issue763.cofit_upload] progress uploads: {len(uploaded)} dirs + raw completions")
        return 0

    if args.directions_only:
        uploaded = _upload_dirs(_DIRECTIONS_ARTIFACTS)
        print(f"[issue763.cofit_upload] directions uploads: {uploaded}")
        return 0

    # FINAL: re-run both passes (idempotent bulk commits) + the deliverables.
    uploaded = _upload_dirs(_PROGRESS_ARTIFACTS + _DIRECTIONS_ARTIFACTS + _FINAL_ARTIFACTS)
    raw = _upload_neutral_raw_completions()
    results = COFIT_DIR / "cofit_results.json"
    nonlinear = COFIT_DIR / "nonlinear_tests.json"
    manifest = COFIT_DIR / "inputs_manifest.json"
    # neutral_arm_manifest is a plan-required §6.5 deliverable: the figures
    # phase stages it from HF on a fresh Phase-C lane; its absence here means
    # the Phase-B assembly (or its upload) never ran (review r1 C2).
    neutral_arm = COFIT_DIR / "neutral_arm_manifest.json"
    missing = [str(p) for p in (results, nonlinear, manifest, neutral_arm) if not p.exists()]
    if missing:
        raise RuntimeError(
            f"final upload: primary deliverables missing {missing} — the epm:results "
            "sentinel must not fire before every §6.5 deliverable exists"
        )
    write_sentinel(
        "epm:results",
        {
            "round": "neutral-contrast-and-cofit",
            "primary_deliverables": [
                "eval_results/issue_763/neutral-contrast-and-cofit/cofit_results.json",
                "eval_results/issue_763/neutral-contrast-and-cofit/nonlinear_tests.json",
                "eval_results/issue_763/neutral-contrast-and-cofit/pv_directions_v2/",
            ],
            "uploaded": uploaded + ([raw] if raw else []),
            "final": True,
        },
    )
    print(f"[issue763.cofit_upload] FINAL uploads: {len(uploaded)} dirs; epm:results emitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
