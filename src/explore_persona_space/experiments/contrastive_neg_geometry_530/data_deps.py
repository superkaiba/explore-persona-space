"""Task #530 — auto-download of #472 carry-over data dependencies.

The #530 pipeline reuses #472's pre-built persona bank, layer centroids,
and on-policy R generations. These artifacts live under
``data/issue_472/`` locally but are **gitignored** (the ``data/`` tree +
``*.pt`` patterns are excluded from version control), so a fresh
worktree on the local VM or a freshly-provisioned pod does not have
them. The bootstrap path on the pod also does not pull them
automatically — the parent #504 line manually downloaded these on the
pod, but #530 entrypoints are run before the pod is provisioned (Phase
0.5 happens on the pod immediately after bootstrap), so they MUST land
on disk before the first script touches them.

This module exposes :func:`prepare_data_dependencies`, an idempotent
helper that pulls the required files from the HF datasets repo at a
**pinned revision** (the same revision ``scripts/i477_reval_confirm.py``
uses, so #530 inherits a bit-for-bit consistent geometry/R bank). The
helper is safe to call from every #530 entrypoint that touches the
data; it skips downloads when local files already exist (sized > 0).

Per-file ``hf_hub_download`` (NOT ``snapshot_download``) is used to
avoid the siblings-truncation gotcha
(``feedback_snapshot_download_siblings_truncation`` memory). The HF
cache lands at the HF default; we copy each file into the rig's
expected local layout (flat ``data/issue_472/`` for ``persona_bank.json``
+ centroids; ``data/issue_472/on_policy_R/`` for the R bundles), which
mirrors the layout #472/#504 originally wrote and that all #530 CLI
defaults assume.

The HF source paths are ``geometry/persona_bank.json`` +
``geometry/centroids_L{10,15,20}.pt`` + ``on_policy_R/R_{train,eval}.json``
under the prefix ``issue472_neg_geometry/``. We copy them into a flat
``data/issue_472/`` layout (i.e. drop the ``geometry/`` subdir locally),
matching the way #472's writer originally laid them out and that every
#504/#530 script reads back.

Fail-loud contract: any unexpected HF error propagates; ``token=None``
is fine (the data repo is public-read) but ``HF_TOKEN`` from ``.env`` is
used when present to avoid rate-limit issues. If a download silently
returns 0 bytes, the size assertion in :func:`_pull_one_file` raises.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

# ── Pinned constants ─────────────────────────────────────────────────────────
# The same revision used by ``scripts/i477_reval_confirm.py`` so #530 inherits
# the exact geometry + R bank that #472/#477 measured against. Do NOT bump
# without a deliberate cross-task consistency check.
DATA_REPO: str = "superkaiba1/explore-persona-space-data"
DATA_REVISION: str = "66d7db7a542e19275f8c1d8e32948396d050faa9"
DATA_PREFIX: str = "issue472_neg_geometry"

# HF-side paths under the prefix (note: ``geometry/`` subdir on HF — we flatten
# locally to match the layout the rig's scripts read).
HF_FILES: tuple[str, ...] = (
    f"{DATA_PREFIX}/geometry/persona_bank.json",
    f"{DATA_PREFIX}/geometry/centroids_L10.pt",
    f"{DATA_PREFIX}/geometry/centroids_L15.pt",
    f"{DATA_PREFIX}/geometry/centroids_L20.pt",
    f"{DATA_PREFIX}/on_policy_R/R_train.json",
    f"{DATA_PREFIX}/on_policy_R/R_eval.json",
)

# Local mirror layout (flat — drop the ``geometry/`` subdir).
LOCAL_DATA_ROOT: Path = Path("data/issue_472")
LOCAL_PATHS: dict[str, Path] = {
    f"{DATA_PREFIX}/geometry/persona_bank.json": LOCAL_DATA_ROOT / "persona_bank.json",
    f"{DATA_PREFIX}/geometry/centroids_L10.pt": LOCAL_DATA_ROOT / "centroids_L10.pt",
    f"{DATA_PREFIX}/geometry/centroids_L15.pt": LOCAL_DATA_ROOT / "centroids_L15.pt",
    f"{DATA_PREFIX}/geometry/centroids_L20.pt": LOCAL_DATA_ROOT / "centroids_L20.pt",
    f"{DATA_PREFIX}/on_policy_R/R_train.json": LOCAL_DATA_ROOT / "on_policy_R" / "R_train.json",
    f"{DATA_PREFIX}/on_policy_R/R_eval.json": LOCAL_DATA_ROOT / "on_policy_R" / "R_eval.json",
}


def _pull_one_file(hf_path: str, local_path: Path, *, token: str | None) -> None:
    """Download one file at the pinned revision and copy into the rig's layout.

    Idempotent: skips when ``local_path`` exists with positive size. Uses
    per-file ``hf_hub_download`` (NOT ``snapshot_download``) to avoid the
    siblings-truncation gotcha.

    Fail-loud: any HF error propagates; a 0-byte landing raises
    ``RuntimeError`` (silent empty download is exactly the failure mode the
    upstream gotcha file warns about).
    """

    from huggingface_hub import hf_hub_download

    if local_path.exists() and local_path.stat().st_size > 0:
        log.info("data_deps: already local: %s (%d bytes)", local_path, local_path.stat().st_size)
        return

    log.info("data_deps: pulling %s @ %s → %s", hf_path, DATA_REVISION[:8], local_path)
    cached = hf_hub_download(
        repo_id=DATA_REPO,
        repo_type="dataset",
        revision=DATA_REVISION,
        filename=hf_path,
        token=token,
    )
    local_path.parent.mkdir(parents=True, exist_ok=True)
    # ``shutil.copyfile`` is idempotent across reruns AND survives the
    # cross-filesystem case (HF cache may sit on a different mount than the
    # repo's ``data/`` tree, in which case ``os.link`` would raise EXDEV).
    shutil.copyfile(cached, local_path)
    size = local_path.stat().st_size
    if size <= 0:
        raise RuntimeError(
            f"data_deps: copied {hf_path} → {local_path} but landed file is empty "
            "(0 bytes). Refusing to proceed — re-run after investigating HF Hub."
        )
    log.info("data_deps: landed %s (%d bytes)", local_path, size)


def prepare_data_dependencies(
    *,
    token: str | None = None,
    files: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """Idempotently download the #472 carry-over artifacts the #530 rig needs.

    Args:
        token: HF token. When ``None``, falls back to ``HF_TOKEN``/``HF_HUB_TOKEN``
            in the environment, then to anonymous access (the data repo is
            public-read).
        files: Subset of ``HF_FILES`` to pull. Defaults to ALL.

    Returns:
        Mapping from HF path → local path for every file requested.
    """
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")

    LOCAL_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    (LOCAL_DATA_ROOT / "on_policy_R").mkdir(parents=True, exist_ok=True)

    targets = files if files is not None else HF_FILES
    for hf_path in targets:
        if hf_path not in LOCAL_PATHS:
            raise ValueError(
                f"data_deps: unknown HF path {hf_path!r}; expected one of {list(LOCAL_PATHS)}"
            )
        _pull_one_file(hf_path, LOCAL_PATHS[hf_path], token=token)

    return {hf_path: LOCAL_PATHS[hf_path] for hf_path in targets}
