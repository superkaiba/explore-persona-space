#!/usr/bin/env python3
"""Stage the scoped #1092 operator-read subset from HF (accelerators OFF, serial).

Downloads exactly the 2 cells x 5 kinds x 3 layers summary .npy + the pinned
corpus manifest into data/issue_1092_inline_operator/. Idempotent (skips files
already present with the expected size). Serialized + bounded retries because a
Hub 429 queue-saturation episode is ongoing.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "0")
# #658/#911: the shared VM root disk (/) is at 100%; route ALL Hub IO onto the
# /mnt/eps-data data disk (60 GB free) so staging never touches `/`.
os.environ.setdefault("HF_HOME", "/mnt/eps-data/thomasjiralerspong/.hf_i1092_operator")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

from huggingface_hub import hf_hub_download, list_repo_tree  # noqa: E402

REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue1092_realistic_crossing"
SUMM_PREFIX = f"{HF_PREFIX}/analysis_tensors/summaries"
CORPUS_REV = "7ef5523673d64697ab497577dbc5b9270c39f020"
SUMM_REV = "main"
STAGE_ROOT = Path("/mnt/eps-data/thomasjiralerspong/issue_1092_inline_operator")
STAGE = STAGE_ROOT / HF_PREFIX

CELLS = ["cell_inst_own", "cell_pre_own"]
KINDS = ("prefix_end", "context_end", "t1", "t2", "t3")
LAYERS = (14, 18, 19)


def _retry(fn, what: str, tries: int = 6):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            wait = min(120, 15 * (i + 1))
            print(
                f"[stage] {what} attempt {i + 1}/{tries} failed: "
                f"{type(e).__name__}: {str(e)[:140]} -> sleep {wait}s",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError(f"[stage] exhausted retries for {what}")


def _stage_one(rel_path: str, revision: str) -> None:
    dest = STAGE_ROOT / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[stage] skip (present) {rel_path} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)
        return
    t0 = time.monotonic()
    # local_dir download lands the file DIRECTLY under STAGE_ROOT (no shared-cache
    # blob on `/`); HF_HOME is also on /mnt/eps-data so lock/etag files stay there.
    _retry(
        lambda: hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=rel_path,
            revision=revision,
            local_dir=str(STAGE_ROOT),
        ),
        rel_path,
    )
    print(
        f"[stage] got {rel_path} ({dest.stat().st_size / 1e6:.1f} MB, "
        f"{time.monotonic() - t0:.0f}s)",
        flush=True,
    )


def main() -> None:
    want_tokens = {f"{k}_L{layer:02d}" for k in KINDS for layer in LAYERS}
    n_files = 0
    for cell in CELLS:
        tree = _retry(
            lambda cell=cell: list(
                list_repo_tree(
                    REPO,
                    repo_type="dataset",
                    revision=SUMM_REV,
                    path_in_repo=f"{SUMM_PREFIX}/{cell}",
                    recursive=True,
                )
            ),
            f"list {cell}",
        )
        for t in tree:
            size = getattr(t, "size", None)
            if size is None:
                continue
            name = t.path.rsplit("/", 1)[-1]
            if any(
                name.startswith(tok + ".") or name.startswith(tok + "_shard") for tok in want_tokens
            ):
                _stage_one(t.path, SUMM_REV)
                n_files += 1
    _stage_one(f"{HF_PREFIX}/corpus/manifest.jsonl", CORPUS_REV)
    print(f"[stage] DONE: {n_files} summary files + manifest under {STAGE}", flush=True)


if __name__ == "__main__":
    main()
