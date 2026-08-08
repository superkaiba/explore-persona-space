#!/usr/bin/env python
"""Issue #1689 user-slot-recapture — mirror the Phase-C fits outputs to HF.

The fits battery (`scripts/issue1689_user_slot_fits.py`) writes its eval JSONs
+ per-cell artifacts locally and does NOT self-upload; on the cpu-bigmem GCE
lane the instance self-DELETEs at exit, so a clean run would lose everything
(the crash-persist path fires only on rc != 0). This mirror step runs as the
LAST link of the workload chain (`fits && mirror`), uploading the round's
`eval_results/issue_1689/` outputs to the HF data repo in ONE `upload_folder`
commit and verifying the uploaded set server-side. Same shape as
`scripts/issue1345_boundary_ablation_stage_and_mirror.py::cmd_mirror`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    assert (repo_root / "scripts" / "issue1689_user_slot_fits.py").exists(), repo_root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

from huggingface_hub import HfApi  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

DATA_REPO = "superkaiba1/explore-persona-space-data"
# The fits battery's own out-dirs ONLY (its --out-dir + --percell-dir
# defaults) — never the whole eval_results/issue_1689 tree, which carries
# every prior round's committed outputs (~5k files of redundant commit).
MIRROR_DIRS = (
    REPO_ROOT / "eval_results" / "issue_1689" / "user_slot_recapture",
    REPO_ROOT / "eval_results" / "issue_1689" / "percell",
)
HF_EVAL_MIRROR_PREFIX = "issue1689_speaker_lattice/user_slot_recapture/eval_mirror"


def main() -> int:
    """Upload the fits battery's out-dirs to the eval_mirror prefix + verify."""
    api = HfApi()
    mirrored_any = False
    for local_dir in MIRROR_DIRS:
        if not local_dir.exists():
            print(f"[fits-mirror] skip absent {local_dir}", flush=True)
            continue
        n_local = sum(1 for p in local_dir.rglob("*") if p.is_file())
        if n_local == 0:
            print(f"[fits-mirror] skip empty {local_dir}", flush=True)
            continue
        mirrored_any = True
        prefix = f"{HF_EVAL_MIRROR_PREFIX}/{local_dir.name}"
        # Deterministic pre-upload guard (outside the transient-retry wrapper):
        # the Hub rejects >10k files per repo dir at COMMIT time, non-retriably
        # (#658).
        hub.assert_hub_dir_filecounts(folder_path=str(local_dir), path_in_repo=prefix)
        hub.retry_transient(
            lambda d=local_dir, p=prefix: api.upload_folder(
                folder_path=str(d),
                repo_id=DATA_REPO,
                repo_type="dataset",
                path_in_repo=p,
            ),
            what=f"upload_folder({prefix})",
        )
        listed = hub.retry_transient(
            lambda p=prefix: hub.list_hf_files_under_path(api, DATA_REPO, p, repo_type="dataset"),
            what=f"verify({prefix})",
        )
        n_remote = len(listed)
        print(f"[fits-mirror] {local_dir.name}: local={n_local} remote={n_remote}", flush=True)
        assert n_remote >= n_local, f"mirror verify FAILED ({prefix}): {n_remote} < {n_local}"
    assert mirrored_any, "no fits outputs to mirror — every MIRROR_DIR absent/empty"
    print("[fits-mirror] verify PASS", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
