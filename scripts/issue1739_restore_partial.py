"""Restore crash-persisted partial fits artifacts for issue #1739 (leg-2 resume staging).

A crashed fits lane's completed-cell artifacts are crash-persisted to the HF
data repo under ``issue1739_partial/<attempt-id>/eval_results_issue_1739/``
(the GCP EXIT-trap layout; e.g. the sycophancy lane att-20260729-032734-syc,
478 files). This script scoped-stages that subtree back onto the local
``eval_results/issue_1739/`` tree BEFORE the fits phase runs, so
``arms.run_grid_multi``'s per-cell resume predicate
(``<behavior>/arm_results/percell/cells.jsonl`` + ``percell/preds/*.npz``)
SKIPs the already-completed cells instead of recomputing ~25 h of fits.

Scoped staging only: ONE server-side ``list_hf_files_under_path`` enumeration
per behavior on the attempt prefix + per-file atomic ``hub.stage_hub_file``
(idempotent: an existing local file is never overwritten, so a re-run — or a
tree the dispatcher already partially wrote — is safe). NEVER
``snapshot_download`` / a bare full listing against the ~1M-file data repo
(gotchas #833). One revision is resolved up front so every file comes from one
snapshot. Invoked from ``scripts/issue1739_leg2.sh`` step 5 when
``EPM_I1739_RESUME_PARTIAL_PREFIX`` is set; behavior scoping follows the
lane's ``EPM_I1739_BEHAVIORS`` convention.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Force the PLAIN download path for THIS script only (must precede the hub /
# huggingface_hub import): the 458-small-npz restore storm WEDGES xet_get
# indefinitely (att-20260730-055211-syc, py-spy-confirmed) and errors
# hf_transfer (att-20260730-063858-syc), while the plain path handles small
# files fine. The disables are deliberately NOT set process-wide by the
# caller: leg2 step 2's >50 GB store tars REQUIRE an accelerator (plain
# download refuses them — att-20260730-065438-syc).
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from explore_persona_space.orchestrate import hub  # noqa: E402  (dotenv before hub import)

PARTIAL_RESULTS_DIRNAME = "eval_results_issue_1739"  # GCP crash-persist layout


def restore_behavior(
    api,
    repo: str,
    prefix: str,
    behavior_root: Path,
    *,
    revision: str,
    max_files: int = 0,
) -> tuple[int, int]:
    """Stage every file under ``prefix`` onto ``behavior_root`` (skip-if-exists).

    Returns ``(staged, already_present)``. ``max_files > 0`` caps the staged
    file COUNT (smoke mode). Empty listing returns ``(0, 0)`` — the caller
    decides whether an all-behaviors-empty restore is fatal.
    """
    files = hub.list_hf_files_under_path(api, repo, prefix, repo_type="dataset", revision=revision)
    if max_files and len(files) > max_files:
        files = sorted(files)[:max_files]
        print(f"[leg2-restore] max-files cap active: staging {len(files)} files", flush=True)
    if not files:
        return 0, 0

    def _one(f: str) -> bool:
        target = behavior_root / Path(f).relative_to(prefix)
        existed = target.exists()
        hub.stage_hub_file(repo, f, target, repo_type="dataset", revision=revision)
        return not existed

    with ThreadPoolExecutor(max_workers=min(6, len(files))) as pool:
        fresh = list(pool.map(_one, files))  # stage_hub_file raises on failure — fail-loud
    return sum(fresh), len(fresh) - sum(fresh)


def main(argv: list[str] | None = None) -> int:
    """CLI: restore ``--hf-prefix``'s per-behavior fits artifacts; exit 1 on an empty prefix."""
    ap = argparse.ArgumentParser(
        description="Restore issue1739_partial crash-persisted fits artifacts (resume staging).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--hf-prefix",
        required=True,
        help="HF attempt prefix, e.g. issue1739_partial/att-20260729-032734-syc",
    )
    ap.add_argument(
        "--behaviors", required=True, help="space-separated behaviors (EPM_I1739_BEHAVIORS scoping)"
    )
    ap.add_argument("--results-root", default="eval_results/issue_1739")
    ap.add_argument("--repo", default=hub.DEFAULT_DATASET_REPO)
    ap.add_argument(
        "--max-files", type=int, default=0, help="cap staged files per behavior (smoke); 0 = no cap"
    )
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi

    api = HfApi()
    info = hub.retry_transient(
        lambda: api.repo_info(args.repo, repo_type="dataset"), what=f"repo_info({args.repo})"
    )
    revision = str(info.sha)
    results_root = Path(args.results_root)
    prefix_root = args.hf_prefix.rstrip("/") + "/" + PARTIAL_RESULTS_DIRNAME
    total = 0
    for behavior in args.behaviors.split():
        prefix = f"{prefix_root}/{behavior}"
        staged, present = restore_behavior(
            api,
            args.repo,
            prefix,
            results_root / behavior,
            revision=revision,
            max_files=args.max_files,
        )
        total += staged + present
        # Verify the restored layout against what the fits resume actually
        # reads: run_grid_multi loads <out_root>/arm_results/percell/cells.jsonl
        # (out_root = <results-root>/<behavior>) and keys SKIPs on its rows.
        cells = results_root / behavior / "arm_results" / "percell" / "cells.jsonl"
        n_rows = 0
        if cells.exists():
            with cells.open(encoding="utf-8") as fh:
                n_rows = sum(1 for line in fh if line.strip())
        preds_dir = results_root / behavior / "arm_results" / "percell" / "preds"
        n_preds = len(list(preds_dir.glob("*.npz"))) if preds_dir.is_dir() else 0
        print(
            f"[leg2-restore] {behavior}: staged={staged} already-present={present} "
            f"resume-rows={n_rows} preds-npz={n_preds}",
            flush=True,
        )
        if (staged + present) and n_rows == 0:
            print(
                f"[leg2-restore] WARNING {behavior}: files restored but {cells} is absent/empty "
                "— the fits resume predicate will skip NOTHING",
                flush=True,
            )
    if total == 0:
        print(
            f"[leg2-restore] FATAL: 0 files under {args.repo}:{prefix_root}/<behavior> "
            "for every requested behavior — wrong --hf-prefix?",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(
        f"[leg2-restore] done: {total} file(s) restored/verified revision={revision[:12]}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
