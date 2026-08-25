#!/usr/bin/env python3
"""Issue #2378 causal-patching-arms — VM-side stage-back of the pod harvest.

Downloads the round's HF raw-completions prefix into the judge/analysis
--patch-root layout (anchors/grid/confirm/bank/meta rollout JSONLs). The
orchestrator-owned stage-back step the r18 review accepted as m9: both VM
legs fail loud on absence, this script is what makes them present. Small
text-only pull (~306 files, tens of MB) — VM staging is in-policy.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue2378_common as cm  # noqa: E402

PREFIX = f"{cm.HF_PREFIX}/raw_completions/causal_patching"


def main() -> int:
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    dest = cm.REPO_ROOT / "data" / "issue_2378" / "patch_round"
    from huggingface_hub import HfApi

    api = HfApi()
    files = hub.list_hf_files_under_path(api, cm.HF_DATA_REPO, PREFIX, repo_type="dataset")
    if not files:
        raise RuntimeError(f"empty stage-back listing under {PREFIX} (fail loud)")
    n = 0
    for path in sorted(files):
        rel = path[len(PREFIX) + 1 :]
        target = dest / rel
        if target.exists():
            continue
        got = hub.retry_transient(
            lambda p=path: hf_hub_download(
                cm.HF_DATA_REPO, p, repo_type="dataset", local_dir="/tmp/i2378_stageback"
            ),
            what=f"download {path}",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(got).read_bytes())
        n += 1
    print(f"[stage-back] {n} downloaded, {len(files) - n} already present -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
