"""Regression test for scripts/sync_models.py::cmd_pull large-repo enumeration.

The HF Hub's ``repo_info().siblings`` list silently truncates at ~7901
entries. ``snapshot_download`` only falls back to the complete
``list_repo_tree`` walk when ``len(siblings) > VERY_LARGE_REPO_THRESHOLD``
(50000), so for a repo whose true file count sits between ~7901 and 50000
(the project model repo carries 20k+ files but reports ~7.4k siblings) the
fallback never fires and a pull of any sub-path past the truncation cap
resolves to ``Fetching 0 files`` even though the files are present.

``cmd_pull`` therefore must NOT rely on ``snapshot_download``'s enumeration.
It enumerates the complete file list via ``list_repo_files_complete`` (which
drives ``list_repo_tree(recursive=True)`` directly, no truncation) and
downloads each matched file by explicit path with ``hf_hub_download``.

These tests mock the HF client to return >7901 entries where the requested
sub-path lives entirely beyond the truncation point, and assert that:
1. ``cmd_pull`` resolves the matching files from the COMPLETE list.
2. It downloads exactly those files via ``hf_hub_download`` (per-file,
   no repo enumeration), never ``snapshot_download``.
3. A genuinely absent sub-path fails loud (SystemExit) rather than
   silently reporting a no-op pull as success.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from huggingface_hub.hf_api import RepoFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# sync_models.py does `from _bootstrap import bootstrap`, which is relative to
# scripts/, so scripts/ must be importable before the module is exec'd.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

_spec = importlib.util.spec_from_file_location(
    "sync_models",
    SCRIPTS_DIR / "sync_models.py",
)
sync_models = importlib.util.module_from_spec(_spec)
sys.modules["sync_models"] = sync_models
_spec.loader.exec_module(sync_models)


# A truncation point well below the real file count, mirroring the ~7901 cap.
TRUNCATION = 7901


def _build_repo_tree(target_dir: str, n_target_files: int = 7) -> list[RepoFile]:
    """Build a repo tree where ``target_dir`` lives ENTIRELY beyond the cap.

    Returns RepoFile entries for ~10k files: the first TRUNCATION are filler
    directories (what a truncated siblings list would surface), and the
    target directory's files are placed after the cap so they would be
    invisible to any siblings-based enumeration.
    """
    entries: list[RepoFile] = []
    # Filler files up to (and a bit past) the truncation point.
    for i in range(TRUNCATION + 2000):
        entries.append(
            RepoFile(
                path=f"filler_dir_{i}/adapter_model.safetensors",
                size=1,
                blob_id="b",
                oid="o",
            )
        )
    # The requested directory, entirely past the cap.
    for j in range(n_target_files):
        entries.append(
            RepoFile(
                path=f"{target_dir}/file_{j}.bin",
                size=1,
                blob_id="b",
                oid="o",
            )
        )
    return entries


def _make_api(tree: list[RepoFile]):
    """A fake HfApi whose list_repo_tree yields the COMPLETE tree.

    Mirrors huggingface_hub 0.36.2, where list_repo_tree(recursive=True) is
    the non-truncating enumeration. The test never exercises the truncated
    siblings path because cmd_pull is required to avoid it.
    """

    class FakeApi:
        def __init__(self, *_args, **_kwargs):
            pass

        def list_repo_tree(self, *_args, **_kwargs):
            return list(tree)

    return FakeApi


def test_cmd_pull_resolves_files_past_truncation(tmp_path):
    """Sub-path beyond the ~7901 cap must still resolve all its files."""
    target = "c_issue376_marker_install_em_seed42_post_em"
    tree = _build_repo_tree(target, n_target_files=7)

    downloaded: list[str] = []

    def fake_hf_hub_download(*, repo_id, filename, repo_type, local_dir, token):
        downloaded.append(filename)
        return str(Path(local_dir) / filename)

    args = SimpleNamespace(
        pull=target, dest=str(tmp_path), repo="superkaiba1/explore-persona-space"
    )

    with (
        patch.dict("os.environ", {"HF_TOKEN": "test_token"}),
        patch("huggingface_hub.HfApi", _make_api(tree)),
        patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download),
        # snapshot_download must NOT be called; blow up loudly if it is.
        patch(
            "huggingface_hub.snapshot_download",
            side_effect=AssertionError("cmd_pull must not call snapshot_download"),
        ),
    ):
        sync_models.cmd_pull(args)

    expected = sorted(f"{target}/file_{j}.bin" for j in range(7))
    assert sorted(downloaded) == expected, (
        f"Expected all 7 target files past the truncation cap, got {downloaded}"
    )


def test_cmd_pull_trailing_slash_is_normalized(tmp_path):
    """A trailing slash on the requested path must not break matching."""
    target = "benign_first/benign_sft_lora_seed42"
    tree = _build_repo_tree(target, n_target_files=3)

    downloaded: list[str] = []

    def fake_hf_hub_download(*, repo_id, filename, repo_type, local_dir, token):
        downloaded.append(filename)
        return str(Path(local_dir) / filename)

    args = SimpleNamespace(
        pull=target + "/", dest=str(tmp_path), repo="superkaiba1/explore-persona-space"
    )

    with (
        patch.dict("os.environ", {"HF_TOKEN": "test_token"}),
        patch("huggingface_hub.HfApi", _make_api(tree)),
        patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download),
    ):
        sync_models.cmd_pull(args)

    assert len(downloaded) == 3
    assert all(f.startswith(target + "/") for f in downloaded)


def test_cmd_pull_absent_path_fails_loud(tmp_path):
    """A genuinely absent sub-path must exit non-zero, never a silent no-op."""
    tree = _build_repo_tree("some_present_dir", n_target_files=4)

    args = SimpleNamespace(
        pull="does_not_exist_anywhere", dest=str(tmp_path), repo="superkaiba1/explore-persona-space"
    )

    with (
        patch.dict("os.environ", {"HF_TOKEN": "test_token"}),
        patch("huggingface_hub.HfApi", _make_api(tree)),
        patch(
            "huggingface_hub.hf_hub_download",
            side_effect=AssertionError("nothing should download for an absent path"),
        ),
        pytest.raises(SystemExit) as exc,
    ):
        sync_models.cmd_pull(args)

    assert exc.value.code == 1
