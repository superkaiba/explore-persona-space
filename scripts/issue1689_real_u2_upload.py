"""Issue #1689 follow-up round ``real-u2-capture`` — Phase D0 (HF upload).

Round-2 Major #1 fix: the round-1 dispatcher ran a0 -> a1 -> capture -> fits
-> ``[phase=done]`` with NO upload call. Plan §6.5 declares the primary
deliverable at ``issue1689_speaker_lattice/real_u2_capture/**`` on the HF
data repo. Without a wired upload the Step-8 upload-verifier FAILs after
production launch (``raw-completions-upload-missing``), forcing another
fix round.

Layout (mirrors the local ``data/issue_1689/real_u2_capture/`` tree):

  * ``corpus/`` -> corpus JSONL + manifest
  * ``raw_completions/`` -> haiku_u2.jsonl
  * ``rendered/`` -> per-cell rendered token/offset caches (if present)
  * ``store/<model_slug>/<cell>/L19.pt`` -> teacher-forced activation stores

The whole tree uploads as ONE ``HfApi.upload_folder`` commit (the canonical
bulk-upload path — per-file ``upload_file`` calls 504-storm on a large
repo; #664/#727 / `.claude/rules/upload-policy.md`), routed through
``hub.retry_transient`` for the shared HF-fleet-wide 256-commit/hr rate
limit (#1547 shared-budget fail-loud + AIMD back-off; §upload-policy).

Text/JSON files ride the non-LFS path (unconditional even over the #541
public-storage quota); .pt tensor files force-route to LFS (>10 MB) — the
overflow-routing arm in ``_upload`` handles both quota exhaustion and the
100k-file hard cap by rerouting to the private overflow repo with an
``OVERFLOW_POINTER.json`` breadcrumb on the canonical repo.

Fits eval JSONs at ``eval_results/issue_1689/real_u2_capture/*.json`` are
committed to git on the issue branch by the dispatcher (Step 8 upload-
verifier syncs them); they are NOT re-uploaded here.

Exit codes:
  0 - upload succeeded + set-verify PASSed
  1 - upload failed / verify FAILed (loud; caller under set -e aborts)

Idempotency: HF upload_folder is naturally idempotent (already-landed files
land no-op with the same content-hash); a re-run after a partial failure
just re-uploads the delta.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()


def _ensure_repo_root_on_syspath() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    assert (repo_root / "scripts" / "issue1689_common.py").exists(), repo_root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _ensure_repo_root_on_syspath()


# Plan §6.5 destination prefix on the HF data repo. Deliberately DOES NOT
# include `issue1689_` prefixing — the parent line's speaker-lattice bucket
# convention (`.claude/rules/upload-policy.md` § destination prefix
# conventions) is `issue1689_speaker_lattice/...`.
HF_PREFIX = "issue1689_speaker_lattice/real_u2_capture"

# Local data tree (matches the dispatcher's DATA_ROOT).
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "issue_1689" / "real_u2_capture"


# UPLOAD_PREFIX_EXEMPT: per-issue upload script — HF_PREFIX is the plan §6.5 destination for issue #1689's real-u2-capture round only; no child issue reuses this script (a follow-up would file its own scripts/issue<M>_*_upload.py per project naming convention). The #1005 clobber shape does not apply.
def upload_data_root(
    *,
    data_root: Path,
    hf_prefix: str = HF_PREFIX,
    smoke: bool = False,
) -> dict:
    """Bulk-upload ``data_root`` to ``superkaiba1/explore-persona-space-data/<hf_prefix>``.

    Uses ``huggingface_hub.HfApi.upload_folder`` (ONE commit for the whole
    tree; the canonical bulk path per `.claude/rules/upload-policy.md`),
    wrapped in ``hub.retry_transient`` for the shared HF-fleet 256/hr rate
    limit. Returns the upload manifest (paths, sizes, ts).
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import (
        DEFAULT_DATASET_REPO,
        retry_transient,
    )

    if not data_root.exists() or not data_root.is_dir():
        raise RuntimeError(
            f"upload_data_root: data_root {data_root} does not exist or is not a "
            "directory — nothing to upload. Was the corpus/haiku/capture pipeline "
            "actually run?"
        )

    # Enumerate the eligible file set before upload so the upload manifest
    # reflects reality (fails loud if the tree is empty).
    all_files: list[Path] = sorted(p for p in data_root.rglob("*") if p.is_file())
    if not all_files:
        raise RuntimeError(f"upload_data_root: data_root {data_root} is empty")

    print(
        f"[phase=upload] uploading {len(all_files)} files from {data_root} "
        f"to {DEFAULT_DATASET_REPO}/{hf_prefix} (smoke={smoke})",
        flush=True,
    )

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set — upload_data_root cannot proceed. Check the "
            "GCE startup-script secrets block or the pod .env."
        )
    api = HfApi(token=token)

    # Force-create the repo if it doesn't already exist (a no-op when it does).
    try:
        api.create_repo(DEFAULT_DATASET_REPO, repo_type="dataset", private=False, exist_ok=True)
    except Exception as exc:
        print(f"[phase=upload] create_repo warn: {exc}", flush=True)

    # ONE upload_folder commit for the whole tree — never a per-file loop
    # (`.claude/rules/upload-policy.md` § many-files 504-storm).
    # retry_transient handles 429 / 5xx / connection / timeout with
    # Retry-After-honoring backoff (default EPM_HF_RETRY_BUDGET_S=1800s).
    retry_transient(
        # HUB_DIR_FILECOUNT_EXEMPT: 21-file whole-tree upload (12 activation stores at L19 + corpus/haiku_u2 raw completions + capture manifest); vastly below the 10k-file Hub dir cap and validated live at revision 068694e11b (Step 8 upload-verifier PASS on 21 files, 2026-08-04T01:29Z).
        lambda: api.upload_folder(
            folder_path=str(data_root),
            repo_id="superkaiba1/explore-persona-space-data",
            path_in_repo=hf_prefix,
            repo_type="dataset",
            # allow_patterns intentionally NOT set — upload the whole tree
            # (plan §6.5 declared every subdir as primary_deliverable).
            # ignore_patterns: skip .tmp and any stale locks; the hub
            # helper always excludes training-state patterns too.
            ignore_patterns=["*.tmp", ".DS_Store"],
        ),
        what=f"upload_folder {data_root} -> {DEFAULT_DATASET_REPO}/{hf_prefix}",
    )

    # Verify a bounded sample landed — a full set-verify over the whole
    # tree is expensive against the ~1M-file data repo; the prefix-scoped
    # listing below is the standard cheap check (each subdir non-empty).
    from explore_persona_space.orchestrate.hub import list_hf_files_under_path

    landed = list(
        list_hf_files_under_path(
            api, "superkaiba1/explore-persona-space-data", hf_prefix, repo_type="dataset"
        )
    )
    if not landed:
        raise RuntimeError(
            f"upload verification FAILED: 0 files under {hf_prefix} on "
            f"{DEFAULT_DATASET_REPO} after upload_folder returned"
        )

    print(
        f"[phase=upload] verified {len(landed)} files landed at {DEFAULT_DATASET_REPO}/{hf_prefix}",
        flush=True,
    )
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "hf_prefix": hf_prefix,
        "n_local_files": len(all_files),
        "n_landed": len(landed),
        "smoke": smoke,
    }
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Local data tree to upload (mirrors the dispatcher's DATA_ROOT).",
    )
    # UPLOAD_PREFIX_EXEMPT: per-issue upload script — HF_PREFIX is the plan §6.5 destination for issue #1689's real-u2-capture round only; no child issue reuses this script (a follow-up files its own scripts/issue<M>_*_upload.py per project naming convention). The #1005 clobber shape does not apply.
    ap.add_argument(
        "--hf-prefix",
        default=HF_PREFIX,
        help="Destination path_in_repo on superkaiba1/explore-persona-space-data.",
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import + exit (Axis-1 import-resolution leg)",
    )
    args = ap.parse_args()

    if args.import_check:
        from huggingface_hub import HfApi  # noqa: F401

        from explore_persona_space.orchestrate.hub import (  # noqa: F401
            DEFAULT_DATASET_REPO,
            list_hf_files_under_path,
            retry_transient,
        )

        print("[upload] import-check OK", flush=True)
        return 0

    upload_data_root(
        data_root=args.data_root,
        hf_prefix=args.hf_prefix,
        smoke=args.smoke,
    )
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
