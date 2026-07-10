# ruff: noqa: E402
"""Pre-upload per-directory file-count guard in ``orchestrate/hub.py`` (#1190).

The HF Hub rejects any single repo directory holding >10,000 files at COMMIT
time — a NON-retriable ``BadRequestError`` fired AFTER the full compute has
run and every byte is staged (#658 r2: 12,000 rollout files staged into one
dir). ``assert_hub_dir_filecounts`` pre-counts the staged files per TARGET
repo directory and raises ``HubDirFileCountError`` BEFORE any network I/O;
``hub._upload`` (folder branch) and ``hub._upload_folder_filtered`` are wired
to it (before ``HfApi`` construction, OUTSIDE the try blocks).

These tests execute the REAL guard + wiring bodies (the #906
one-production-body rule): the only fake is the external HfApi network
boundary, and the fake's method ``def``s mirror the exact call shapes hub.py
uses (signature-conformant by construction).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import huggingface_hub
import pytest

from explore_persona_space.orchestrate import hub

HUB_LOGGER = "explore_persona_space.orchestrate.hub"


def _make_tree(root: Path, subdir: str, n: int, ext: str = ".json") -> Path:
    """Create ``n`` empty files under ``root/subdir`` and return ``root``."""
    d = root / subdir
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"f{i:05d}{ext}").touch()
    return root


@pytest.fixture(autouse=True)
def _guard_env(monkeypatch):
    """Hermetic default: guard ON (kill switch unset)."""
    monkeypatch.delenv("EPM_SKIP_HF_DIR_FILECOUNT_GUARD", raising=False)


@pytest.fixture(scope="module")
def big_tree(tmp_path_factory) -> Path:
    """A real-scale tree: 10,001 empty files in ONE directory (``flat/``).

    Module-scoped — creation costs ~1-2 s; shared by the real-scale counting,
    message-content, and wiring tests below.
    """
    root = tmp_path_factory.mktemp("issue1190_big")
    return _make_tree(root, "flat", hub.HUB_DIR_FILE_LIMIT + 1)


# ---------------------------------------------------------------------------
# count_staged_files_per_repo_dir — counting / filtering / prefix mapping
# ---------------------------------------------------------------------------


class TestCounting:
    def test_counts_keyed_by_target_repo_dir(self, tmp_path):
        """Counts key on path_in_repo prefix + relative subdir, not local dir."""
        _make_tree(tmp_path, "a", 3)
        _make_tree(tmp_path, "a/b", 2)
        (tmp_path / "root.json").touch()
        counts = hub.count_staged_files_per_repo_dir(tmp_path, "issueX/raw")
        assert counts == {"issueX/raw/a": 3, "issueX/raw/a/b": 2, "issueX/raw": 1}, counts

    def test_empty_prefix_maps_to_relative_dirs(self, tmp_path):
        """path_in_repo='' keys on the bare relative subdir ('' for root files)."""
        _make_tree(tmp_path, "a", 2)
        (tmp_path / "root.json").touch()
        counts = hub.count_staged_files_per_repo_dir(tmp_path, "")
        assert counts == {"a": 2, "": 1}, counts

    def test_allow_patterns_count_only_staged_subset(self, tmp_path):
        """Only files upload_folder would actually stage are counted."""
        _make_tree(tmp_path, "a", 4, ext=".json")
        _make_tree(tmp_path, "a", 3, ext=".pt")  # adds 3 .pt beside the 4 .json
        counts = hub.count_staged_files_per_repo_dir(tmp_path, "p", allow_patterns=["**/*.json"])
        assert counts == {"p/a": 4}, counts

    def test_ignore_patterns_excluded(self, tmp_path):
        _make_tree(tmp_path, "a", 4, ext=".json")
        _make_tree(tmp_path, "a", 3, ext=".pt")
        counts = hub.count_staged_files_per_repo_dir(tmp_path, "p", ignore_patterns=["*.pt"])
        assert counts == {"p/a": 4}, counts

    def test_default_ignore_patterns_exclude_git(self, tmp_path):
        """Parity with upload_folder's own default excludes (.git/ etc.)."""
        _make_tree(tmp_path, "a", 2)
        _make_tree(tmp_path, ".git", 5, ext="")
        counts = hub.count_staged_files_per_repo_dir(tmp_path, "p")
        assert counts == {"p/a": 2}, counts

    def test_real_scale_counting_under_5s(self, big_tree):
        """Counting a 10,001-file tree completes well under the 5 s bound
        (measured ~1.25 s at this scale on the dev VM — negligible next to
        actually uploading such a tree; loose bound to avoid CI flake)."""
        t0 = time.monotonic()
        counts = hub.count_staged_files_per_repo_dir(big_tree, "dest")
        dt = time.monotonic() - t0
        assert counts == {"dest/flat": hub.HUB_DIR_FILE_LIMIT + 1}, counts
        assert dt < 5.0, f"counting took {dt:.2f}s (bound 5s)"


# ---------------------------------------------------------------------------
# assert_hub_dir_filecounts — boundaries, message, kill switch
# ---------------------------------------------------------------------------


class TestBoundaries:
    def test_exactly_limit_does_not_raise(self, tmp_path):
        """Strict > : exactly-limit is server-legal ("up to 10000")."""
        _make_tree(tmp_path, "a", 7)
        counts = hub.assert_hub_dir_filecounts(tmp_path, "p", limit=7, warn_at=100)
        assert counts == {"p/a": 7}, counts

    def test_limit_plus_one_raises(self, tmp_path):
        """An >= implementation must fail this pair of tests."""
        _make_tree(tmp_path, "a", 8)
        with pytest.raises(hub.HubDirFileCountError):
            hub.assert_hub_dir_filecounts(tmp_path, "p", limit=7, warn_at=100)

    def test_exactly_warn_at_stays_silent(self, tmp_path, caplog):
        _make_tree(tmp_path, "a", 5)
        with caplog.at_level(logging.WARNING, logger=HUB_LOGGER):
            hub.assert_hub_dir_filecounts(tmp_path, "p", limit=100, warn_at=5)
        assert caplog.records == [], [r.getMessage() for r in caplog.records]

    def test_warn_at_plus_one_warns(self, tmp_path, caplog):
        _make_tree(tmp_path, "a", 6)
        with caplog.at_level(logging.WARNING, logger=HUB_LOGGER):
            counts = hub.assert_hub_dir_filecounts(tmp_path, "p", limit=100, warn_at=5)
        assert counts == {"p/a": 6}, counts
        warned = "\n".join(r.getMessage() for r in caplog.records)
        assert "recommended shard size" in warned, warned
        assert "5" in warned and "100" in warned, warned


class TestRaiseMessage:
    def test_message_names_dir_counts_recipe_and_switches(self, big_tree):
        """The raise message is the actionable artifact: it must name the
        offending dir, the comma-formatted staged count + cap, the shard
        recipe, the gotchas.md pointer, the kill switch, and the
        outside-retry-wrapper clause."""
        with pytest.raises(hub.HubDirFileCountError) as exc_info:
            hub.assert_hub_dir_filecounts(big_tree, "dest")
        msg = str(exc_info.value)
        assert "'dest/flat'" in msg, msg
        assert "10,001" in msg, msg  # comma-formatted staged count
        assert "10,000" in msg, msg  # comma-formatted cap
        assert "shard_NNNN/" in msg, msg
        assert "5,000" in msg, msg  # comma-formatted shard-recipe size
        assert ".claude/rules/gotchas.md" in msg, msg
        assert "EPM_SKIP_HF_DIR_FILECOUNT_GUARD=1" in msg, msg
        assert "OUTSIDE any transient-retry wrapper" in msg, msg
        assert "#658" in msg, msg

    def test_message_disjoint_from_transient_error_scan(self, big_tree):
        """A bare "5000"/"10000" contains "500", which the response-less scan
        in ``_is_transient_upload_error`` reads as an HTTP 500 — a retry
        wrapper would then RETRY the deterministic guard failure. Pin the
        disjointness BOTH ways: no "500"-family substring survives once the
        comma-formatted numbers are removed, AND the live predicate itself
        classifies the exception as non-transient."""
        with pytest.raises(hub.HubDirFileCountError) as exc_info:
            hub.assert_hub_dir_filecounts(big_tree, "dest")
        msg = str(exc_info.value)
        scrubbed = msg.replace("10,001", "").replace("10,000", "").replace("5,000", "")
        for code in ("500", "502", "503", "504"):
            assert code not in scrubbed, f"bare {code!r} in guard message: {msg}"
        assert not hub._is_transient_upload_error(exc_info.value), (
            f"guard raise classified TRANSIENT — a retry wrapper would retry it: {msg}"
        )


class TestKillSwitch:
    def test_kill_switch_degrades_to_logged_warning(self, tmp_path, monkeypatch, caplog):
        """EPM_SKIP_HF_DIR_FILECOUNT_GUARD=1: no raise, WARNING actually
        logged (not a silent skip), and the per-dir counts still returned."""
        monkeypatch.setenv("EPM_SKIP_HF_DIR_FILECOUNT_GUARD", "1")
        _make_tree(tmp_path, "a", 9)
        with caplog.at_level(logging.WARNING, logger=HUB_LOGGER):
            counts = hub.assert_hub_dir_filecounts(tmp_path, "p", limit=8, warn_at=4)
        assert counts == {"p/a": 9}, counts
        warned = "\n".join(r.getMessage() for r in caplog.records)
        assert "EPM_SKIP_HF_DIR_FILECOUNT_GUARD=1 set" in warned, warned
        assert "'p/a'" in warned, warned

    def test_guard_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("EPM_SKIP_HF_DIR_FILECOUNT_GUARD", raising=False)
        assert hub._dir_filecount_guard_enabled() is True
        monkeypatch.setenv("EPM_SKIP_HF_DIR_FILECOUNT_GUARD", "1")
        assert hub._dir_filecount_guard_enabled() is False


# ---------------------------------------------------------------------------
# Wiring — _upload folder branch + _upload_folder_filtered raise BEFORE any
# HfApi construction / method call (network fully mocked)
# ---------------------------------------------------------------------------


class FakeHfApi:
    """Signature-conformant HfApi fake for the network boundary.

    Records construction (``constructed``) and every method call (``calls``)
    so the wiring tests can assert the guard raise fires BEFORE any network
    object exists. Method ``def``s mirror hub.py's exact call shapes.
    """

    def __init__(self):
        self.constructed = 0
        self.calls: list[tuple[str, str, str]] = []

    # factory shim: hub code calls huggingface_hub.HfApi(token=...)
    def __call__(self, token=None):
        self.constructed += 1
        return self

    def create_repo(self, repo_id, *, repo_type=None, private=False, exist_ok=False):
        self.calls.append(("create_repo", repo_id, f"private={private}"))

    def upload_folder(
        self,
        *,
        folder_path,
        repo_id,
        path_in_repo,
        repo_type,
        allow_patterns=None,
        ignore_patterns=None,
    ):
        self.calls.append(("upload_folder", repo_id, path_in_repo))

    def upload_file(self, *, path_or_fileobj, repo_id, path_in_repo, repo_type):
        self.calls.append(("upload_file", repo_id, path_in_repo))

    def list_repo_tree(
        self, *, repo_id, repo_type=None, revision=None, recursive=False, path_in_repo=None
    ):
        self.calls.append(("list_repo_tree", repo_id, str(path_in_repo)))
        return iter([])

    def file_exists(self, repo_id, path, *, repo_type=None, revision=None):
        self.calls.append(("file_exists", repo_id, path))
        return True


@pytest.fixture
def fake_api(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.setenv("EPM_HF_STORAGE_CHECK", "0")  # keep any headroom probes offline
    api = FakeHfApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", api)
    return api


class TestWiring:
    def test_upload_folder_branch_raises_on_over_limit_dir(self, big_tree, fake_api):
        """_upload's folder branch raises HubDirFileCountError BEFORE HfApi is
        even constructed — zero network I/O, and the raise propagates (it is
        NOT swallowed into the `except Exception -> return ""` path)."""
        with pytest.raises(hub.HubDirFileCountError):
            hub._upload(big_tree, "user/repo", "dataset", "dest")
        assert fake_api.constructed == 0, fake_api.calls
        assert fake_api.calls == [], fake_api.calls

    def test_upload_folder_filtered_raises_on_over_limit_dir(self, big_tree, fake_api):
        """_upload_folder_filtered raises BEFORE HfApi construction when the
        allow_patterns-selected subset is over the cap."""
        with pytest.raises(hub.HubDirFileCountError):
            hub._upload_folder_filtered(
                big_tree,
                "user/repo",
                "dataset",
                "dest",
                allow_patterns=["**/*.json"],
                expected_repo_paths=["dest/flat/f00000.json"],
            )
        assert fake_api.constructed == 0, fake_api.calls
        assert fake_api.calls == [], fake_api.calls

    def test_upload_folder_filtered_counts_only_staged_subset(self, big_tree, fake_api):
        """Acceptance #4: an allow_patterns subset UNDER the cap proceeds past
        the guard even though the local dir holds >10k files — the guard
        counts what upload_folder will actually stage, not the local dir."""
        result = hub._upload_folder_filtered(
            big_tree,
            "user/repo",
            "dataset",
            "dest",
            allow_patterns=["flat/f00000.json"],
            expected_repo_paths=["dest/flat/f00000.json"],
        )
        # The guard let it through; the (mocked) upload ran. The empty mocked
        # listing then fails the exact-set verify -> "" (existing contract).
        assert fake_api.constructed == 1
        assert ("upload_folder", "user/repo", "dest") in fake_api.calls
        assert result == ""

    def test_upload_normal_small_tree_inert(self, tmp_path, fake_api):
        """Normal-scale uploads are untouched: the guard is inert and the
        existing _upload flow proceeds to the (mocked) network."""
        _make_tree(tmp_path, "a", 3)
        result = hub._upload(tmp_path, "user/repo", "dataset", "dest")
        assert fake_api.constructed == 1
        assert ("upload_folder", "user/repo", "dest") in fake_api.calls
        # Empty mocked listing -> verification returns "" (existing contract).
        assert result == ""

    def test_upload_kill_switch_proceeds_with_warning(
        self, big_tree, fake_api, monkeypatch, caplog
    ):
        """With the kill switch set, the over-limit upload proceeds to the
        (mocked) network and the degrade-WARNING is logged."""
        monkeypatch.setenv("EPM_SKIP_HF_DIR_FILECOUNT_GUARD", "1")
        with caplog.at_level(logging.WARNING, logger=HUB_LOGGER):
            hub._upload(big_tree, "user/repo", "dataset", "dest")
        assert fake_api.constructed == 1
        assert ("upload_folder", "user/repo", "dest") in fake_api.calls
        warned = "\n".join(r.getMessage() for r in caplog.records)
        assert "EPM_SKIP_HF_DIR_FILECOUNT_GUARD=1 set" in warned, warned
