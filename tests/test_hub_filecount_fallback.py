# ruff: noqa: E402
"""Reactive file-count-limit overflow fallback in ``hub`` (#1108 → #2304).

The canonical HF model repo hit the 100,000-files-per-repo hard limit (#1090:
"Your git repo would contain 100050 files after this push, over the limit of
100000 files") and the DATA repo hit the same rejection at 1,000,000 files
(#2162). The shared gate helper ``_filecount_overflow_retry`` now retries a
REJECTED model- OR dataset-repo upload — single-file/dir via ``_upload`` AND
bulk via ``_upload_folder_filtered`` — against the private overflow repo
(``DEFAULT_OVERFLOW_REPO``), emits the #564 routing event with
``reason="file-count-limit-reactive"``, writes the ``OVERFLOW_POINTER.json``
breadcrumb on the canonical repo (typed ``repo_type``; degrading EXPLICITLY to
``"unwritable-filecount-cap"`` when the canonical repo refuses the pointer at
its own cap), and records an observed-count sentinel row for EVERY enabled
file-count refusal (reroutable or not — the both-repos-at-cap row included).
Default ON, kill switch ``EPM_HF_FILECOUNT_FALLBACK=0`` (zero side effects
when off, sentinel writes included).

These tests execute the REAL production bodies end-to-end (the #906
one-production-body rule): ``_upload``, ``_upload_folder_filtered``,
``_filecount_overflow_retry``, ``_write_overflow_pointer``,
``upload_raw_completions_to_data_repo``, and the sentinel helpers all run
for real. The only fake is the external HfApi network boundary, and the
fake's method ``def``s mirror the exact call shapes hub.py uses
(signature-conformant by construction — a drifted call site raises
TypeError instead of silently passing).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import huggingface_hub
import pytest
from huggingface_hub.hf_api import RepoFile

from explore_persona_space.orchestrate import hub

# The verbatim #1090 rejection (full format confirmed by HF forum thread 26400).
FILECOUNT_MSG = (
    "Your git repo would contain 100050 files after this push, over the limit of 100000 files."
)


class _HubRejection(Exception):
    """Message-carrying stand-in for the (unverified-class) HF rejection."""

    def __init__(self, msg: str, status_code: int | None = None):
        super().__init__(msg)
        if status_code is not None:
            # Shape mirrors requests' err.response.status_code read in
            # hub._is_transient_upload_error.
            class _Resp:
                pass

            resp = _Resp()
            resp.status_code = status_code
            resp.headers = {}
            self.response = resp


def _rf(path: str) -> RepoFile:
    return RepoFile(path=path, size=1, blob_id="b", oid="o")


class FakeHfApi:
    """Signature-conformant HfApi fake for the hub-upload network boundary.

    ``folder_errors`` maps repo_id -> Exception raised by ``upload_folder``;
    ``file_errors`` maps repo_id -> Exception raised by ``upload_file`` (#2304
    — the pointer-degradation + single-file-refusal seams); ``tree`` maps
    repo_id -> RepoFile list returned by the scoped verify walk. Every call is
    recorded in ``calls`` as ``(method, repo_id, detail)``; ``upload_file``
    calls are ALSO recorded (before any injected raise) in ``file_uploads`` as
    ``(repo_id, path_in_repo, repo_type)`` so tests can assert the pointer's
    repo_type without changing ``calls``'s 3-tuple shape.
    """

    def __init__(self):
        self.folder_errors: dict[str, Exception] = {}
        self.file_errors: dict[str, Exception] = {}
        self.tree: dict[str, list[RepoFile]] = {}
        self.calls: list[tuple[str, str, str]] = []
        self.file_uploads: list[tuple[str, str, str]] = []

    # --- factory shim: hub code calls huggingface_hub.HfApi(token=...) ---
    def __call__(self, token=None):
        return self

    # --- HfApi surface (defs mirror hub.py's exact call shapes) ---
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
        err = self.folder_errors.get(repo_id)
        if err is not None:
            raise err

    def upload_file(self, *, path_or_fileobj, repo_id, path_in_repo, repo_type):
        self.calls.append(("upload_file", repo_id, path_in_repo))
        self.file_uploads.append((repo_id, path_in_repo, repo_type))
        err = self.file_errors.get(repo_id)
        if err is not None:
            raise err

    def list_repo_tree(
        self, *, repo_id, repo_type=None, revision=None, recursive=False, path_in_repo=None
    ):
        self.calls.append(("list_repo_tree", repo_id, str(path_in_repo)))
        return iter(self.tree.get(repo_id, []))

    def file_exists(self, repo_id, path, *, repo_type=None, revision=None):
        self.calls.append(("file_exists", repo_id, path))
        return False

    # --- assertion helpers ---
    def n_calls(self, method: str, repo_id: str) -> int:
        return sum(1 for m, r, _ in self.calls if m == method and r == repo_id)


DEST = "adapters/issue1108_test/final_adapter"


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """Local dir + fake api + hermetic env (no network, event + sentinel sinks in tmp)."""
    src = tmp_path / "adapter_dir"
    src.mkdir()
    (src / "adapter_model.safetensors").write_bytes(b"\x00" * 16)
    event_path = tmp_path / "overflow-events.jsonl"
    monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.setenv("EPM_HF_STORAGE_CHECK", "0")  # emitters probe headroom: keep offline
    monkeypatch.setenv("EPM_HF_OVERFLOW_EVENT_PATH", str(event_path))
    # #2304: the fallback now writes observed-count sentinel rows — keep them
    # hermetic (never ~/.cache) so parallel test sessions cannot cross-talk.
    monkeypatch.setenv("EPM_HF_FILECOUNT_SENTINEL_PATH", str(tmp_path / "filecount.jsonl"))
    monkeypatch.delenv("EPM_HF_FILECOUNT_FALLBACK", raising=False)
    api = FakeHfApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", api)
    return src, api, event_path


def _events(event_path: Path) -> list[dict]:
    if not event_path.exists():
        return []
    return [json.loads(line) for line in event_path.open(encoding="utf-8") if line.strip()]


def _sentinel_rows() -> list[dict]:
    """Rows of the observed-count sentinel (resolved via the PRODUCTION path
    resolver, so the test also exercises the env-override branch)."""
    path = hub._filecount_sentinel_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


class TestReroute:
    def test_filecount_rejection_reroutes_event_and_pointer(self, rig):
        src, api, event_path = rig
        api.folder_errors[hub.DEFAULT_MODEL_REPO] = _HubRejection(FILECOUNT_MSG)
        api.tree[hub.DEFAULT_OVERFLOW_REPO] = [_rf(f"{DEST}/adapter_model.safetensors")]

        result = hub._upload(src, hub.DEFAULT_MODEL_REPO, "model", DEST)

        assert result == f"{hub.DEFAULT_OVERFLOW_REPO}/{DEST}"
        # overflow repo created PRIVATE, canonical attempted first
        assert ("create_repo", hub.DEFAULT_OVERFLOW_REPO, "private=True") in api.calls
        assert api.n_calls("upload_folder", hub.DEFAULT_MODEL_REPO) == 1
        assert api.n_calls("upload_folder", hub.DEFAULT_OVERFLOW_REPO) == 1
        # routing event carries the reactive reason
        rows = _events(event_path)
        assert len(rows) == 1, rows
        assert rows[0]["reason"] == "file-count-limit-reactive"
        assert rows[0]["original_repo"] == hub.DEFAULT_MODEL_REPO
        assert rows[0]["effective_repo"] == hub.DEFAULT_OVERFLOW_REPO
        # pointer breadcrumb attempted against the CANONICAL repo
        pointer_calls = [
            (m, r, p)
            for m, r, p in api.calls
            if m == "upload_file" and p.endswith("OVERFLOW_POINTER.json")
        ]
        assert pointer_calls == [
            ("upload_file", hub.DEFAULT_MODEL_REPO, f"{DEST}/OVERFLOW_POINTER.json")
        ]

    def test_kill_switch_off_restores_legacy_empty_return(self, rig, monkeypatch):
        src, api, event_path = rig
        monkeypatch.setenv("EPM_HF_FILECOUNT_FALLBACK", "0")
        api.folder_errors[hub.DEFAULT_MODEL_REPO] = _HubRejection(FILECOUNT_MSG)

        result = hub._upload(src, hub.DEFAULT_MODEL_REPO, "model", DEST)

        assert result == ""
        assert api.n_calls("upload_folder", hub.DEFAULT_OVERFLOW_REPO) == 0
        assert _events(event_path) == []

    def test_overflow_repo_rejection_no_recursion(self, rig):
        """The guard short-circuits when the target IS the overflow repo."""
        src, api, event_path = rig
        api.folder_errors[hub.DEFAULT_OVERFLOW_REPO] = _HubRejection(FILECOUNT_MSG)

        result = hub._upload(src, hub.DEFAULT_OVERFLOW_REPO, "model", DEST)

        assert result == ""
        assert api.n_calls("upload_folder", hub.DEFAULT_OVERFLOW_REPO) == 1  # no recursion
        assert _events(event_path) == []

    def test_dataset_repo_type_rerouted(self, rig):
        """#2304 pin FLIP: the old ``test_dataset_repo_type_not_rerouted``
        pinned the #1108 model-only scope this task removes by design — a
        dataset-repo file-count refusal now reroutes exactly like a model one,
        with a dataset-typed pointer + a sentinel observation row."""
        src, api, event_path = rig
        api.folder_errors[hub.DEFAULT_DATASET_REPO] = _HubRejection(FILECOUNT_MSG)
        api.tree[hub.DEFAULT_OVERFLOW_REPO] = [_rf("bucket/x/adapter_model.safetensors")]

        result = hub._upload(src, hub.DEFAULT_DATASET_REPO, "dataset", "bucket/x")

        assert result == f"{hub.DEFAULT_OVERFLOW_REPO}/bucket/x"
        assert ("create_repo", hub.DEFAULT_OVERFLOW_REPO, "private=True") in api.calls
        assert api.n_calls("upload_folder", hub.DEFAULT_DATASET_REPO) == 1
        assert api.n_calls("upload_folder", hub.DEFAULT_OVERFLOW_REPO) == 1
        rows = _events(event_path)
        assert len(rows) == 1, rows
        assert rows[0]["reason"] == "file-count-limit-reactive"
        assert rows[0]["original_repo"] == hub.DEFAULT_DATASET_REPO
        assert rows[0]["effective_repo"] == hub.DEFAULT_OVERFLOW_REPO
        # pointer breadcrumb against the CANONICAL dataset repo, dataset-typed
        assert (
            hub.DEFAULT_DATASET_REPO,
            "bucket/x/OVERFLOW_POINTER.json",
            "dataset",
        ) in api.file_uploads
        # sentinel: exactly one blocked observation, for the canonical repo
        srows = _sentinel_rows()
        assert [(r["repo_id"], r["repo_type"], r["status"]) for r in srows] == [
            (hub.DEFAULT_DATASET_REPO, "dataset", "blocked")
        ]
        assert srows[0]["observed_files"] == 100050 and srows[0]["limit"] == 100000


class TestNonMatchingErrors:
    @pytest.mark.parametrize(
        "err",
        [
            _HubRejection("403 Forbidden: You have exceeded your public storage space"),
            # Real HF 4xx rejections carry a response status code, which decides
            # transience BY CODE (no substring scan — "25000" would otherwise
            # false-read as a transient "500", the #989 digit-triplet trap).
            _HubRejection(
                "400 Bad Request: this directory is over the limit of 10000 files per folder",
                status_code=400,
            ),
            _HubRejection(
                "400 Bad Request: a commit cannot contain more than 25000 operations; "
                "split your push",
                status_code=400,
            ),
        ],
        ids=["storage-403", "per-folder-cap", "per-commit-op-cap"],
    )
    def test_non_matching_error_no_reroute(self, rig, err):
        src, api, event_path = rig
        api.folder_errors[hub.DEFAULT_MODEL_REPO] = err

        result = hub._upload(src, hub.DEFAULT_MODEL_REPO, "model", DEST)

        assert result == ""
        assert api.n_calls("upload_folder", hub.DEFAULT_OVERFLOW_REPO) == 0
        assert _events(event_path) == []

    def test_transient_504_exhausts_then_no_reroute(self, rig, monkeypatch):
        src, api, event_path = rig
        monkeypatch.setenv("EPM_HF_RETRY_BUDGET_S", "0")  # attempt-bound only
        monkeypatch.setattr(hub.time, "sleep", lambda s: None)
        api.folder_errors[hub.DEFAULT_MODEL_REPO] = _HubRejection(
            "504 Gateway Time-out", status_code=504
        )

        result = hub._upload(src, hub.DEFAULT_MODEL_REPO, "model", DEST)

        assert result == ""
        assert api.n_calls("upload_folder", hub.DEFAULT_MODEL_REPO) > 1  # retried
        assert api.n_calls("upload_folder", hub.DEFAULT_OVERFLOW_REPO) == 0
        assert _events(event_path) == []


class TestDeleteAfter:
    def test_local_reaped_only_after_verified_overflow_landing(self, rig):
        src, api, _ = rig
        api.folder_errors[hub.DEFAULT_MODEL_REPO] = _HubRejection(FILECOUNT_MSG)
        api.tree[hub.DEFAULT_OVERFLOW_REPO] = [_rf(f"{DEST}/adapter_model.safetensors")]

        result = hub._upload(src, hub.DEFAULT_MODEL_REPO, "model", DEST, delete_after=True)

        assert result == f"{hub.DEFAULT_OVERFLOW_REPO}/{DEST}"
        assert not src.exists()  # reaped by the RECURSIVE call, post-verify

    def test_overflow_verify_miss_keeps_local_intact(self, rig):
        """Overflow upload 'succeeds' but the scoped verify finds 0 files:
        no reap, no event, no pointer, empty return."""
        src, api, event_path = rig
        api.folder_errors[hub.DEFAULT_MODEL_REPO] = _HubRejection(FILECOUNT_MSG)
        api.tree[hub.DEFAULT_OVERFLOW_REPO] = []  # verify walk resolves nothing

        result = hub._upload(src, hub.DEFAULT_MODEL_REPO, "model", DEST, delete_after=True)

        assert result == ""
        assert src.exists() and (src / "adapter_model.safetensors").exists()
        assert _events(event_path) == []
        assert not any(p.endswith("OVERFLOW_POINTER.json") for _, _, p in api.calls)

    def test_overflow_upload_failure_keeps_local_intact(self, rig):
        src, api, event_path = rig
        api.folder_errors[hub.DEFAULT_MODEL_REPO] = _HubRejection(FILECOUNT_MSG)
        api.folder_errors[hub.DEFAULT_OVERFLOW_REPO] = _HubRejection("500 upstream exploded")

        # 500 is transient: keep the retry loop fast + attempt-bound.
        import unittest.mock as _m

        with _m.patch.object(hub.time, "sleep", lambda s: None):
            import os

            os.environ["EPM_HF_RETRY_BUDGET_S"] = "0"
            try:
                result = hub._upload(src, hub.DEFAULT_MODEL_REPO, "model", DEST, delete_after=True)
            finally:
                os.environ.pop("EPM_HF_RETRY_BUDGET_S", None)

        assert result == ""
        assert src.exists()
        assert _events(event_path) == []


class TestPredicates:
    @pytest.mark.parametrize(
        ("msg", "expected"),
        [
            (FILECOUNT_MSG, True),
            # lowercase / prefix-wrapped variants of the same server phrase
            (
                "BadRequest: your git repo would contain 100001 files after this push, "
                "over the limit of 100000 files",
                True,
            ),
            ("429 too many requests", False),
            ("403 Forbidden: You have exceeded your public storage space", False),
            # digit-triplet paths must not read as a limit rejection (#989 family)
            ("upload of issue504_raw/part100000.json failed: connection reset", False),
            # per-FOLDER cap: has "over the limit of"+"files" but no "push"
            ("this directory is over the limit of 10000 files per folder", False),
            # per-commit operation cap: has "push" but not "over the limit of"
            ("a commit cannot contain more than 25000 operations; split your push", False),
        ],
    )
    def test_is_file_count_limit_error(self, msg, expected):
        assert hub._is_file_count_limit_error(Exception(msg)) is expected

    def test_fallback_enabled_default_on(self, monkeypatch):
        monkeypatch.delenv("EPM_HF_FILECOUNT_FALLBACK", raising=False)
        assert hub._filecount_fallback_enabled() is True
        monkeypatch.setenv("EPM_HF_FILECOUNT_FALLBACK", "0")
        assert hub._filecount_fallback_enabled() is False
        monkeypatch.setenv("EPM_HF_FILECOUNT_FALLBACK", "1")
        assert hub._filecount_fallback_enabled() is True


# --------------------------------------------------------------------------
# #2304 — dataset single-file path + recursion bound on the dataset type
# --------------------------------------------------------------------------


class TestDatasetSingleFile:
    def test_dataset_single_file_reroutes_with_dataset_pointer(self, rig, tmp_path):
        """Single-file dataset refusal reroutes; the pointer ATTEMPT is
        dataset-typed against the canonical repo — and because the canonical
        repo is at its cap, the pointer itself degrades EXPLICITLY (the same
        pair of event rows the #2304 live probe expects)."""
        _, api, event_path = rig
        one = tmp_path / "one.json"
        one.write_text("{}")
        api.file_errors[hub.DEFAULT_DATASET_REPO] = _HubRejection(FILECOUNT_MSG)
        api.tree[hub.DEFAULT_OVERFLOW_REPO] = [_rf("bucket/y/one.json")]

        result = hub._upload(
            one, hub.DEFAULT_DATASET_REPO, "dataset", "bucket/y/one.json", upload_as_file=True
        )

        assert result == f"{hub.DEFAULT_OVERFLOW_REPO}/bucket/y/one.json"
        assert api.n_calls("upload_file", hub.DEFAULT_OVERFLOW_REPO) == 1
        reasons = [r["reason"] for r in _events(event_path)]
        assert reasons == [
            "file-count-limit-reactive",
            "overflow-pointer-unwritable-filecount-cap",
        ]
        # the pointer attempt targeted the CANONICAL dataset repo, dataset-typed
        assert (
            hub.DEFAULT_DATASET_REPO,
            "bucket/y/one.json/OVERFLOW_POINTER.json",
            "dataset",
        ) in api.file_uploads
        srows = _sentinel_rows()
        assert [(r["repo_id"], r["status"]) for r in srows] == [
            (hub.DEFAULT_DATASET_REPO, "blocked")
        ]

    def test_overflow_repo_rejection_no_recursion_dataset(self, rig):
        """Dataset variant of the recursion bound: a refusal ON the overflow
        repo is observed (sentinel row) but never rerouted."""
        src, api, event_path = rig
        api.folder_errors[hub.DEFAULT_OVERFLOW_REPO] = _HubRejection(FILECOUNT_MSG)

        result = hub._upload(src, hub.DEFAULT_OVERFLOW_REPO, "dataset", DEST)

        assert result == ""
        assert api.n_calls("upload_folder", hub.DEFAULT_OVERFLOW_REPO) == 1  # no recursion
        assert _events(event_path) == []
        srows = _sentinel_rows()
        assert [(r["repo_id"], r["status"]) for r in srows] == [
            (hub.DEFAULT_OVERFLOW_REPO, "blocked")
        ]


# --------------------------------------------------------------------------
# #2304 — bulk (_upload_folder_filtered) fallback
# --------------------------------------------------------------------------


def _make_eval_tree(tmp_path: Path) -> tuple[Path, list[str]]:
    """A two-cell eval_results tree matching upload_raw_completions' globs."""
    eval_dir = tmp_path / "eval_results" / "issue2304_test"
    for cell in ("cellA", "cellB"):
        (eval_dir / cell).mkdir(parents=True)
        (eval_dir / cell / "raw_completions.json").write_text("{}")
    return eval_dir, ["cellA/raw_completions.json", "cellB/raw_completions.json"]


class TestBulkFallback:
    PIR = "issue2304_test/raw_completions"

    def test_bulk_dataset_rejection_reroutes_and_verifies(self, rig, tmp_path):
        _, api, event_path = rig
        eval_dir, rels = _make_eval_tree(tmp_path)
        expected = [f"{self.PIR}/{rel}" for rel in rels]
        api.folder_errors[hub.DEFAULT_DATASET_REPO] = _HubRejection(FILECOUNT_MSG)
        api.tree[hub.DEFAULT_OVERFLOW_REPO] = [_rf(p) for p in expected]

        result = hub._upload_folder_filtered(
            eval_dir,
            hub.DEFAULT_DATASET_REPO,
            "dataset",
            self.PIR,
            ["raw_completions.json", "**/raw_completions.json"],
            expected,
        )

        assert result == f"{hub.DEFAULT_OVERFLOW_REPO}/{self.PIR}"
        assert ("create_repo", hub.DEFAULT_OVERFLOW_REPO, "private=True") in api.calls
        assert api.n_calls("upload_folder", hub.DEFAULT_DATASET_REPO) == 1
        assert api.n_calls("upload_folder", hub.DEFAULT_OVERFLOW_REPO) == 1
        # the recursive call re-ran the exact-set verify against the OVERFLOW repo
        assert api.n_calls("list_repo_tree", hub.DEFAULT_OVERFLOW_REPO) >= 1
        rows = _events(event_path)
        assert [r["reason"] for r in rows] == ["file-count-limit-reactive"]
        assert rows[0]["original_repo"] == hub.DEFAULT_DATASET_REPO
        # pointer landed on the canonical repo (not itself refused here)
        assert (
            hub.DEFAULT_DATASET_REPO,
            f"{self.PIR}/OVERFLOW_POINTER.json",
            "dataset",
        ) in api.file_uploads
        srows = _sentinel_rows()
        assert [(r["repo_id"], r["status"]) for r in srows] == [
            (hub.DEFAULT_DATASET_REPO, "blocked")
        ]

    def test_raw_completions_mapping_reflects_overflow(self, rig, tmp_path):
        """Production body of upload_raw_completions_to_data_repo end-to-end:
        the returned URL map is built from the bulk upload's OWN base URL, so
        a rerouted tree maps to OVERFLOW paths — never canonical paths that
        hold nothing."""
        _, api, _ = rig
        eval_dir, rels = _make_eval_tree(tmp_path)
        api.folder_errors[hub.DEFAULT_DATASET_REPO] = _HubRejection(FILECOUNT_MSG)
        api.tree[hub.DEFAULT_OVERFLOW_REPO] = [_rf(f"{self.PIR}/{rel}") for rel in rels]

        uploaded = hub.upload_raw_completions_to_data_repo("issue2304_test", eval_dir)

        assert uploaded == {rel: f"{hub.DEFAULT_OVERFLOW_REPO}/{self.PIR}/{rel}" for rel in rels}

    def test_bulk_verify_miss_does_not_reroute(self, rig, tmp_path):
        """Trap 3: the verify-miss ``return ""`` inside the try (an INCOMPLETE
        canonical commit — the server ACCEPTED the push) never triggers the
        fallback."""
        _, api, event_path = rig
        eval_dir, rels = _make_eval_tree(tmp_path)
        expected = [f"{self.PIR}/{rel}" for rel in rels]
        # no folder_errors: the canonical commit "succeeds" but lands nothing
        api.tree[hub.DEFAULT_DATASET_REPO] = []

        result = hub._upload_folder_filtered(
            eval_dir,
            hub.DEFAULT_DATASET_REPO,
            "dataset",
            self.PIR,
            ["raw_completions.json", "**/raw_completions.json"],
            expected,
        )

        assert result == ""
        assert api.n_calls("upload_folder", hub.DEFAULT_OVERFLOW_REPO) == 0
        assert _events(event_path) == []
        assert _sentinel_rows() == []

    def test_bulk_non_matching_error_no_reroute(self, rig, tmp_path):
        _, api, event_path = rig
        eval_dir, rels = _make_eval_tree(tmp_path)
        expected = [f"{self.PIR}/{rel}" for rel in rels]
        api.folder_errors[hub.DEFAULT_DATASET_REPO] = _HubRejection(
            "403 Forbidden: You have exceeded your public storage space"
        )

        result = hub._upload_folder_filtered(
            eval_dir,
            hub.DEFAULT_DATASET_REPO,
            "dataset",
            self.PIR,
            ["raw_completions.json", "**/raw_completions.json"],
            expected,
        )

        assert result == ""
        assert api.n_calls("upload_folder", hub.DEFAULT_OVERFLOW_REPO) == 0
        assert _events(event_path) == []
        assert _sentinel_rows() == []  # message-match gate precedes the observation

    def test_bulk_kill_switch_restores_legacy(self, rig, tmp_path, monkeypatch):
        _, api, event_path = rig
        monkeypatch.setenv("EPM_HF_FILECOUNT_FALLBACK", "0")
        eval_dir, rels = _make_eval_tree(tmp_path)
        expected = [f"{self.PIR}/{rel}" for rel in rels]
        api.folder_errors[hub.DEFAULT_DATASET_REPO] = _HubRejection(FILECOUNT_MSG)

        result = hub._upload_folder_filtered(
            eval_dir,
            hub.DEFAULT_DATASET_REPO,
            "dataset",
            self.PIR,
            ["raw_completions.json", "**/raw_completions.json"],
            expected,
        )

        assert result == ""
        assert api.n_calls("upload_folder", hub.DEFAULT_OVERFLOW_REPO) == 0
        assert _events(event_path) == []
        # kill switch OFF ⇒ ZERO side effects, sentinel writes included
        assert _sentinel_rows() == []
        assert not hub._filecount_sentinel_path().exists()

    def test_both_repos_at_cap_records_overflow_observation(self, rig, tmp_path):
        """§8 row 4's mitigation is REAL: when canonical AND overflow both
        refuse at their caps, a sentinel row lands for the OVERFLOW repo
        (branch order: observation BEFORE the identity conjunct returns None),
        no routing event / pointer is written (nothing landed), and the caller
        still fails loud."""
        _, api, event_path = rig
        eval_dir, _ = _make_eval_tree(tmp_path)
        api.folder_errors[hub.DEFAULT_DATASET_REPO] = _HubRejection(FILECOUNT_MSG)
        api.folder_errors[hub.DEFAULT_OVERFLOW_REPO] = _HubRejection(FILECOUNT_MSG)

        with pytest.raises(RuntimeError, match="bulk folder upload failed"):
            hub.upload_raw_completions_to_data_repo("issue2304_test", eval_dir)

        srows = _sentinel_rows()
        assert [(r["repo_id"], r["repo_type"], r["status"]) for r in srows] == [
            (hub.DEFAULT_DATASET_REPO, "dataset", "blocked"),
            (hub.DEFAULT_OVERFLOW_REPO, "dataset", "blocked"),
        ]
        assert _events(event_path) == []  # nothing landed ⇒ no routing event
        assert api.file_uploads == []  # ⇒ no pointer attempt either


# --------------------------------------------------------------------------
# #2304 — kill switch on the dataset _upload path
# --------------------------------------------------------------------------


class TestDatasetKillSwitch:
    def test_dataset_kill_switch_restores_legacy(self, rig, monkeypatch):
        src, api, event_path = rig
        monkeypatch.setenv("EPM_HF_FILECOUNT_FALLBACK", "0")
        api.folder_errors[hub.DEFAULT_DATASET_REPO] = _HubRejection(FILECOUNT_MSG)

        result = hub._upload(src, hub.DEFAULT_DATASET_REPO, "dataset", "bucket/x")

        assert result == ""
        assert api.n_calls("upload_folder", hub.DEFAULT_OVERFLOW_REPO) == 0
        assert _events(event_path) == []
        assert _sentinel_rows() == []
        assert not hub._filecount_sentinel_path().exists()


# --------------------------------------------------------------------------
# #2304 — pointer degradation statuses + the model-path regression guard
# --------------------------------------------------------------------------


class TestPointerStatuses:
    def test_pointer_unwritable_filecount_cap_logs_distinct_reason_and_event(self, rig):
        _, api, event_path = rig
        api.file_errors[hub.DEFAULT_DATASET_REPO] = _HubRejection(FILECOUNT_MSG)

        status = hub._write_overflow_pointer(
            canonical_repo=hub.DEFAULT_DATASET_REPO,
            path_in_repo="bucket/x",
            overflow_repo=hub.DEFAULT_OVERFLOW_REPO,
            repo_type="dataset",
        )

        assert status == "unwritable-filecount-cap"
        rows = _events(event_path)
        assert len(rows) == 1, rows
        assert rows[0]["reason"] == "overflow-pointer-unwritable-filecount-cap"
        assert rows[0]["original_repo"] == hub.DEFAULT_DATASET_REPO
        assert rows[0]["effective_repo"] == hub.DEFAULT_OVERFLOW_REPO

    def test_pointer_transport_blip_stays_generic(self, rig):
        _, api, event_path = rig
        api.file_errors[hub.DEFAULT_DATASET_REPO] = _HubRejection(
            "403 Forbidden: token lacks write permission"
        )

        status = hub._write_overflow_pointer(
            canonical_repo=hub.DEFAULT_DATASET_REPO,
            path_in_repo="bucket/x",
            overflow_repo=hub.DEFAULT_OVERFLOW_REPO,
            repo_type="dataset",
        )

        assert status == "failed"
        assert _events(event_path) == []  # generic failures get NO event row

    def test_pointer_default_repo_type_is_model(self, rig):
        """Keyword-only default byte-preserves the #564 upload_model call
        shape: a repo_type-less call still targets the model repo class."""
        _, api, _ = rig

        status = hub._write_overflow_pointer(
            canonical_repo=hub.DEFAULT_MODEL_REPO,
            path_in_repo=DEST,
            overflow_repo=hub.DEFAULT_OVERFLOW_REPO,
        )

        assert status == "ok"
        assert (
            hub.DEFAULT_MODEL_REPO,
            f"{DEST}/OVERFLOW_POINTER.json",
            "model",
        ) in api.file_uploads

    def test_model_pointer_repo_type_still_model(self, rig):
        """Acceptance 3 regression guard: the model reroute's pointer is
        model-typed — the #1108 path's routing behavior is unchanged."""
        src, api, _ = rig
        api.folder_errors[hub.DEFAULT_MODEL_REPO] = _HubRejection(FILECOUNT_MSG)
        api.tree[hub.DEFAULT_OVERFLOW_REPO] = [_rf(f"{DEST}/adapter_model.safetensors")]

        result = hub._upload(src, hub.DEFAULT_MODEL_REPO, "model", DEST)

        assert result == f"{hub.DEFAULT_OVERFLOW_REPO}/{DEST}"
        assert (
            hub.DEFAULT_MODEL_REPO,
            f"{DEST}/OVERFLOW_POINTER.json",
            "model",
        ) in api.file_uploads


# --------------------------------------------------------------------------
# #2304 — observed-count sentinel unit + integration behavior
# --------------------------------------------------------------------------


class TestFilecountSentinel:
    def test_observation_row_parses_counts_comma_tolerant(self, rig):
        hub._record_filecount_observation(
            "r/d",
            "dataset",
            Exception(
                "Your git repo would contain 1,000,009 files after this push, "
                "over the limit of 1,000,000 files."
            ),
        )
        (row,) = _sentinel_rows()
        assert row["observed_files"] == 1000009
        assert row["limit"] == 1000000
        assert row["status"] == "blocked"
        assert row["repo_id"] == "r/d" and row["repo_type"] == "dataset"
        assert "over the limit" in row["message_excerpt"]

    def test_observation_unparseable_message_records_none_counts(self, rig):
        hub._record_filecount_observation(
            "r/d",
            "model",
            Exception("a NEW server phrasing: too many files, push over the limit of"),
        )
        (row,) = _sentinel_rows()
        assert row["observed_files"] is None and row["limit"] is None
        assert row["status"] == "blocked"

    def test_recovery_appended_only_on_blocked_to_accepting_transition(self, rig):
        hub._record_filecount_observation("r/d", "dataset", Exception(FILECOUNT_MSG))
        hub._maybe_record_filecount_recovery("r/d", "dataset")
        assert [r["status"] for r in _sentinel_rows()] == ["blocked", "accepting"]
        # a repeated success is silent (last row already "accepting")
        hub._maybe_record_filecount_recovery("r/d", "dataset")
        assert [r["status"] for r in _sentinel_rows()] == ["blocked", "accepting"]

    def test_no_recovery_without_prior_blocked(self, rig):
        hub._maybe_record_filecount_recovery("r/d", "dataset")  # no sentinel file at all
        assert not hub._filecount_sentinel_path().exists()
        # a blocked row for a DIFFERENT (repo, type) key never triggers one
        hub._record_filecount_observation("other/repo", "model", Exception(FILECOUNT_MSG))
        hub._maybe_record_filecount_recovery("r/d", "dataset")
        assert [(r["repo_id"], r["status"]) for r in _sentinel_rows()] == [
            ("other/repo", "blocked")
        ]

    def test_upload_success_after_blocked_appends_recovery(self, rig):
        """Integration: the PRODUCTION success-path wiring inside ``_upload``
        appends the blocked→accepting transition row."""
        src, api, _ = rig
        hub._record_filecount_observation(hub.DEFAULT_MODEL_REPO, "model", Exception(FILECOUNT_MSG))
        api.tree[hub.DEFAULT_MODEL_REPO] = [_rf(f"{DEST}/adapter_model.safetensors")]

        result = hub._upload(src, hub.DEFAULT_MODEL_REPO, "model", DEST)

        assert result == f"{hub.DEFAULT_MODEL_REPO}/{DEST}"
        assert [r["status"] for r in _sentinel_rows()] == ["blocked", "accepting"]
