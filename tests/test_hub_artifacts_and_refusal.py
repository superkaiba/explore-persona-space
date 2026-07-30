"""Tests for the data-eval recurring-fix workstream.

Covers (all network mocked):
  - ``list_repo_files_complete`` paginated enumeration handles a >page-size repo
    (mock yields >7901 entries; assert all are enumerated, folders dropped).
  - ``verify_artifacts_exist`` flags a missing HF/WandB URL and passes when all
    cited artifacts resolve; raises on a malformed plan path.
  - ``detect_refusal`` returns True on a canned refusal and False on a normal
    completion (the Claude judge is mocked).
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.hf_api import RepoFile, RepoFolder
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from explore_persona_space.eval.refusal import detect_refusal, filter_refusals
from explore_persona_space.orchestrate.hub import (
    _HF_URL_RE,
    hf_url_path_has_glob,
    list_hf_files_under_path,
    list_repo_files_complete,
    verify_artifacts_exist,
)

# ── HF helpers ────────────────────────────────────────────────────────────────


def _repo_file(path: str) -> RepoFile:
    return RepoFile(path=path, size=1, blob_id="b", oid="o")


def _repo_folder(path: str) -> RepoFolder:
    return RepoFolder(path=path, tree_id="t", oid="o")


def _make_api_with_files(file_paths):
    """Build a mock HfApi whose list_repo_tree yields the given files + folders.

    Interleaves a RepoFolder between files to confirm folders are dropped.
    """
    api = MagicMock()

    def _tree(*args, **kwargs):
        yield _repo_folder("some_dir")
        for p in file_paths:
            yield _repo_file(p)

    api.list_repo_tree.side_effect = _tree
    return api


class TestListRepoFilesComplete:
    def test_enumerates_more_than_page_size(self):
        """A repo with >7901 entries enumerates fully (no siblings truncation)."""
        n = 7901 + 250  # past the repo_info().siblings truncation cap
        paths = [f"checkpoints/shard_{i:05d}.safetensors" for i in range(n)]
        api = _make_api_with_files(paths)

        result = list_repo_files_complete(api, "owner/repo", repo_type="model")

        assert len(result) == n, "all files past the ~7901 cap must be enumerated"
        assert result == sorted(paths)
        # RepoFolder entries are dropped.
        assert "some_dir" not in result
        # The paginated tree walk was used, recursively.
        assert api.list_repo_tree.call_args.kwargs["recursive"] is True

    def test_folders_excluded(self):
        api = _make_api_with_files(["a.json", "b.json"])
        result = list_repo_files_complete(api, "owner/repo")
        assert result == ["a.json", "b.json"]


# ── list_hf_files_under_path (the shared scoped-listing helper, #988) ────────


def _http_err(code: int, msg: str | None = None) -> HfHubHTTPError:
    """HfHubHTTPError whose .response.status_code == code (mirrors
    tests/test_hub.py's helper) so the classifier's status-code branch is
    exercised as in prod."""
    r = MagicMock()
    r.status_code = code
    return HfHubHTTPError(msg or f"{code} error", response=r)


class _ProbeApi:
    """Signature-mirroring ``HfApi`` stand-in for the exact-file fallback tests.

    Boundary fakes conform BY CONSTRUCTION (def-mirrored ``list_repo_tree`` /
    ``file_exists``), never a bare ``MagicMock`` — the real helper body runs
    end to end against them (code-style rule: one production-body test per
    seam-stubbed function, #906).

    ``file_exists_raises`` scripts transport errors for the #1360 retry tests:
    each ``file_exists`` call pops + raises the next exception in order, then
    later calls return ``file_exists_result``.
    """

    def __init__(
        self,
        *,
        tree_raises: Exception,
        file_exists_result: bool = False,
        file_exists_raises: list[Exception] | None = None,
    ):
        self._tree_raises = tree_raises
        self.file_exists_result = file_exists_result
        self._file_exists_raises = list(file_exists_raises or [])
        self.tree_calls: list[dict] = []
        self.file_exists_calls: list[dict] = []

    def list_repo_tree(
        self, *, repo_id, repo_type=None, revision=None, recursive=False, path_in_repo=None
    ):
        self.tree_calls.append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "revision": revision,
                "recursive": recursive,
                "path_in_repo": path_in_repo,
            }
        )
        raise self._tree_raises

    def file_exists(self, repo_id, filename, *, repo_type=None, revision=None):
        self.file_exists_calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "repo_type": repo_type,
                "revision": revision,
            }
        )
        if self._file_exists_raises:
            raise self._file_exists_raises.pop(0)
        return self.file_exists_result

    def list_repo_files(self, *args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("bare full-repo listing (list_repo_files) must never be called")


class TestListHfFilesUnderPath:
    """Unit tests for the #920/#988 scoped-listing helper (all network mocked)."""

    def test_dir_path_threads_path_in_repo_and_prefix_filters(self):
        """A dir path threads the server-side scope kwarg; the defensive
        client-side filter drops out-of-prefix paths a strict fake (whose
        list_repo_tree ignores path_in_repo) would leak through."""
        api = _make_api_with_files(["a/b/one.json", "a/b/two.json", "other/x.json"])
        result = list_hf_files_under_path(api, "owner/repo", "a/b/", repo_type="dataset")
        assert api.list_repo_tree.call_args.kwargs["path_in_repo"] == "a/b"
        assert result == ["a/b/one.json", "a/b/two.json"]
        api.file_exists.assert_not_called()

    def test_exact_file_falls_back_to_file_exists_true(self):
        """The tree endpoint 404s on an exact FILE path (EntryNotFoundError);
        a True file_exists probe resolves it to [path]."""
        api = _ProbeApi(
            tree_raises=EntryNotFoundError("entry a/b/x.json not found"),
            file_exists_result=True,
        )
        result = list_hf_files_under_path(api, "owner/repo", "a/b/x.json", repo_type="dataset")
        assert result == ["a/b/x.json"]
        assert api.tree_calls[0]["path_in_repo"] == "a/b/x.json"
        assert api.file_exists_calls == [
            {
                "repo_id": "owner/repo",
                "filename": "a/b/x.json",
                "repo_type": "dataset",
                "revision": None,
            }
        ]

    def test_absent_path_returns_empty(self):
        api = _ProbeApi(
            tree_raises=EntryNotFoundError("entry ghost not found"), file_exists_result=False
        )
        assert list_hf_files_under_path(api, "owner/repo", "ghost", repo_type="dataset") == []
        assert len(api.file_exists_calls) == 1

    @pytest.mark.parametrize("bad_path", ["", "/", "//"])
    def test_empty_path_raises_value_error(self, bad_path):
        """A falsy/slash-only path would silently degrade to the full-repo
        listing the helper exists to avoid — it raises instead."""
        api = _ProbeApi(tree_raises=AssertionError("tree walk must not fire on an empty path"))
        with pytest.raises(ValueError, match="empty path"):
            list_hf_files_under_path(api, "owner/repo", bad_path)
        assert api.tree_calls == []
        assert api.file_exists_calls == []

    def test_repository_not_found_propagates(self):
        """Repo-level not-found is NOT mapped to [] — it propagates so callers
        fail loud rather than reading a real artifact as missing."""
        api = _ProbeApi(tree_raises=RepositoryNotFoundError("repo gone"))
        with pytest.raises(RepositoryNotFoundError):
            list_hf_files_under_path(api, "owner/ghost", "a/b")
        assert api.file_exists_calls == []

    def test_exact_file_fallback_retries_transient_429_then_succeeds(self):
        """#1360: the exact-file ``file_exists`` fallback rides ``_retry_upload``
        — a transient HF 429 ('maximum queue size reached', the #1315 p11 kill
        shape) retries instead of propagating to _upload's
        log-and-return-\"\" arm."""
        api = _ProbeApi(
            tree_raises=EntryNotFoundError("entry a/b/x.json not found"),
            file_exists_result=True,
            file_exists_raises=[
                _http_err(429, "429 Client Error: Too Many Requests ... maximum queue size reached")
            ],
        )
        with patch("explore_persona_space.orchestrate.hub.time.sleep") as mock_sleep:
            result = list_hf_files_under_path(api, "owner/repo", "a/b/x.json", repo_type="dataset")
        assert result == ["a/b/x.json"]
        assert len(api.file_exists_calls) == 2
        mock_sleep.assert_called_once()

    def test_exact_file_fallback_exhausts_and_propagates(self, monkeypatch):
        """A PERSISTENT 429 exhausts the attempt floor (wall-clock budget kill
        switch = 0) and re-raises fail-loud after exactly 6 calls (the #735
        contract) — never a silent swallow."""
        monkeypatch.setenv("EPM_HF_RETRY_BUDGET_S", "0")
        err = _http_err(429, "429 Client Error: Too Many Requests")
        api = _ProbeApi(
            tree_raises=EntryNotFoundError("entry ghost not found"),
            file_exists_raises=[err] * 6,
        )
        with (
            patch("explore_persona_space.orchestrate.hub.time.sleep"),
            pytest.raises(HfHubHTTPError),
        ):
            list_hf_files_under_path(api, "owner/repo", "ghost", repo_type="dataset")
        assert len(api.file_exists_calls) == 6

    def test_exact_file_fallback_content_class_immediate_reraise(self):
        """The persistent storage-quota-403 (content-class) re-raises on call 1
        with zero sleeps — the wrap must not delay the #564 overflow-routing /
        fail-loud semantics."""
        api = _ProbeApi(
            tree_raises=EntryNotFoundError("entry a/b/x.json not found"),
            file_exists_raises=[
                _http_err(403, "403 Forbidden: You have exceeded your public storage space")
            ],
        )
        with (
            patch("explore_persona_space.orchestrate.hub.time.sleep") as mock_sleep,
            pytest.raises(HfHubHTTPError),
        ):
            list_hf_files_under_path(api, "owner/repo", "a/b/x.json", repo_type="dataset")
        assert len(api.file_exists_calls) == 1
        mock_sleep.assert_not_called()


# ── verify_artifacts_exist ────────────────────────────────────────────────────


class _RepoTypeFallbackApi:
    """Signature-mirroring ``HfApi`` stand-in for the #1482 repo_type-fallback
    tests: the MODEL endpoint raises ``RepositoryNotFoundError``; the DATASET
    endpoint resolves (or also raises, under ``dataset_exists=False``).

    Boundary fakes conform BY CONSTRUCTION (def-mirrored ``list_repo_tree`` /
    ``repo_info``), never a bare ``MagicMock`` — the real
    ``verify_artifacts_exist`` body runs end to end against them (code-style
    rule: one production-body test per seam-stubbed function, #906).
    """

    def __init__(self, dataset_files=(), *, dataset_exists=True):
        self._dataset_files = list(dataset_files)
        self._dataset_exists = dataset_exists
        self.tree_repo_types: list[str | None] = []
        self.repo_info_repo_types: list[str | None] = []

    def list_repo_tree(
        self, *, repo_id, repo_type=None, revision=None, recursive=False, path_in_repo=None
    ):
        self.tree_repo_types.append(repo_type)
        if repo_type != "dataset" or not self._dataset_exists:
            raise RepositoryNotFoundError(f"404: {repo_id} does not resolve as {repo_type}")
        return [_repo_file(p) for p in self._dataset_files]

    def repo_info(self, repo_id, *, repo_type=None, revision=None):
        self.repo_info_repo_types.append(repo_type)
        if repo_type != "dataset" or not self._dataset_exists:
            raise RepositoryNotFoundError(f"404: {repo_id} does not resolve as {repo_type}")
        return SimpleNamespace(id=repo_id)


class TestVerifyArtifactsExist:
    def test_raises_on_missing_plan_path(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            verify_artifacts_exist(tmp_path / "nope.md")

    def test_raises_on_empty_plan_path(self):
        with pytest.raises(ValueError, match="empty"):
            verify_artifacts_exist("")

    def test_raises_when_plan_path_is_dir(self, tmp_path):
        with pytest.raises(ValueError, match="not a file"):
            verify_artifacts_exist(tmp_path)

    def test_no_urls_returns_ok(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text("This plan cites no carry-over artifacts.\n")
        with patch("huggingface_hub.HfApi"):
            ok, missing = verify_artifacts_exist(plan)
        assert ok is True
        assert missing == []

    def test_all_present_passes(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text(
            "Carry over the adapter at "
            "https://huggingface.co/org/models/tree/main/cond_seed42 "
            "and the dataset at "
            "https://huggingface.co/datasets/org/data/tree/abc/issue1_x "
            "plus the run https://wandb.ai/team/proj/runs/run123 .\n"
        )

        api = _make_api_with_files(["cond_seed42/adapter_model.safetensors", "issue1_x/data.jsonl"])
        with (
            patch("huggingface_hub.HfApi", return_value=api),
            patch(
                "explore_persona_space.orchestrate.hub._wandb_run_exists",
                return_value=True,
            ),
        ):
            ok, missing = verify_artifacts_exist(plan)

        assert ok is True
        assert missing == []

    def test_flags_missing_hf_path(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text(
            "Reuse https://huggingface.co/superkaiba1/explore-persona-space/tree/main/ghost_seed99\n"
        )
        # The repo resolves but the cited path is NOT among its files.
        api = _make_api_with_files(["cond_seed42/adapter_model.safetensors"])
        with patch("huggingface_hub.HfApi", return_value=api):
            ok, missing = verify_artifacts_exist(plan)

        assert ok is False
        assert missing == [
            "https://huggingface.co/superkaiba1/explore-persona-space/tree/main/ghost_seed99"
        ]

    def test_flags_missing_wandb_run(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text("Resume metrics from https://wandb.ai/team/proj/runs/deadbeef\n")
        with (
            patch("huggingface_hub.HfApi"),
            patch(
                "explore_persona_space.orchestrate.hub._wandb_run_exists",
                return_value=False,
            ),
        ):
            ok, missing = verify_artifacts_exist(plan)

        assert ok is False
        assert missing == ["https://wandb.ai/team/proj/runs/deadbeef"]

    def test_hf_uri_form_and_repo_root(self, tmp_path):
        """hf:// form with @revision and a bare repo-root URL both resolve."""
        plan = tmp_path / "plan.md"
        plan.write_text(
            "adapter hf://superkaiba1/explore-persona-space@main/cond_seed42 and the "
            "whole repo https://huggingface.co/superkaiba1/explore-persona-space\n"
        )
        api = _make_api_with_files(["cond_seed42/adapter_model.safetensors"])
        with patch("huggingface_hub.HfApi", return_value=api):
            ok, missing = verify_artifacts_exist(plan)
        assert ok is True
        assert missing == []

    def test_json_blob_trailing_punctuation_not_captured(self, tmp_path):
        """URLs inside a JSON blob (each followed by '",') verify against clean paths.

        Regression for incident #541: the trailing '",' rode into the captured
        revision/path and the existence check probed a wrong path, false-blocking
        the pre-launch gate.
        """
        plan = tmp_path / "plan.md"
        plan.write_text(
            '{"adapter": "https://huggingface.co/org/models/tree/main/cond_seed42",\n'
            ' "data": "https://huggingface.co/datasets/org/data/tree/abc/issue1_x",\n'
            ' "uri": "hf://org/models@main/cond_seed42",\n'
            ' "run": "https://wandb.ai/team/proj/runs/run123"}\n'
        )
        api = _make_api_with_files(["cond_seed42/adapter_model.safetensors", "issue1_x/data.jsonl"])
        with (
            patch("huggingface_hub.HfApi", return_value=api),
            patch(
                "explore_persona_space.orchestrate.hub._wandb_run_exists",
                return_value=True,
            ),
        ):
            ok, missing = verify_artifacts_exist(plan)
        assert ok is True
        assert missing == []

    def test_backtick_wrapped_urls_verify_ok(self, tmp_path):
        """Markdown backtick-wrapped URLs terminate at the closing backtick."""
        plan = tmp_path / "plan.md"
        plan.write_text(
            "Reuse `https://huggingface.co/org/models/tree/main/cond_seed42` and "
            "`hf://org/models@main/cond_seed42` per the parent plan.\n"
        )
        api = _make_api_with_files(["cond_seed42/adapter_model.safetensors"])
        with patch("huggingface_hub.HfApi", return_value=api):
            ok, missing = verify_artifacts_exist(plan)
        assert ok is True
        assert missing == []

    def test_json_blob_missing_artifact_still_fails(self, tmp_path):
        """Punctuation stripping must not weaken the gate: a real phantom still FAILs.

        Also pins that the reported missing URL is the CLEAN url (no trailing '",').
        """
        plan = tmp_path / "plan.md"
        plan.write_text('{"adapter": "https://huggingface.co/org/models/tree/main/ghost_seed99"}\n')
        api = _make_api_with_files(["cond_seed42/adapter_model.safetensors"])
        with patch("huggingface_hub.HfApi", return_value=api):
            ok, missing = verify_artifacts_exist(plan)
        assert ok is False
        assert missing == ["https://huggingface.co/org/models/tree/main/ghost_seed99"]

    def test_json_suffix_path_not_truncated(self, tmp_path):
        """A cited path ending in '.json' keeps its suffix (no '.'-stripping)."""
        plan = tmp_path / "plan.md"
        plan.write_text(
            '"https://huggingface.co/datasets/org/data/blob/main/issue1_x/results.json",\n'
        )
        api = _make_api_with_files(["issue1_x/results.json"])
        with patch("huggingface_hub.HfApi", return_value=api):
            ok, missing = verify_artifacts_exist(plan)
        assert ok is True
        assert missing == []

    def test_repo_root_url_uses_repo_info_never_lists(self, tmp_path):
        """A repo-root URL is proven by ONE repo_info call — never a tree
        listing of any scope (#920/#988)."""
        plan = tmp_path / "plan.md"
        plan.write_text("whole repo https://huggingface.co/org/models\n")
        api = MagicMock()
        with patch("huggingface_hub.HfApi", return_value=api):
            ok, missing = verify_artifacts_exist(plan)
        assert ok is True
        assert missing == []
        assert api.repo_info.call_count == 1
        assert api.list_repo_tree.call_count == 0

    def test_repo_root_neither_repo_type_is_missing_row(self, tmp_path):
        """Acceptance 3, repo-root (empty-path) variant: a bare ``hf://org/name``
        citation whose repo resolves under NEITHER repo_type is a reportable
        ``missing`` row — the #1482 reclassification of the former
        propagate-on-``RepositoryNotFoundError`` behavior (#988 site 1)."""
        plan = tmp_path / "plan.md"
        plan.write_text("whole repo hf://org/ghost-repo\n")
        api = _RepoTypeFallbackApi(dataset_exists=False)
        with patch("huggingface_hub.HfApi", return_value=api):
            ok, missing = verify_artifacts_exist(plan)
        assert ok is False
        assert missing == ["hf://org/ghost-repo"]
        # Empty-path branch: repo_info probed as model, retried as dataset.
        assert api.repo_info_repo_types == ["model", "dataset"]
        assert api.tree_repo_types == []

    def test_cited_dir_path_threads_scoped_walk(self, tmp_path):
        """A cited dir path scopes the tree walk server-side via path_in_repo
        (#920/#988) instead of full-listing the repo."""
        plan = tmp_path / "plan.md"
        plan.write_text("adapter https://huggingface.co/org/models/tree/main/cond_seed42\n")
        api = _make_api_with_files(["cond_seed42/adapter_model.safetensors"])
        with patch("huggingface_hub.HfApi", return_value=api):
            ok, missing = verify_artifacts_exist(plan)
        assert ok is True
        assert missing == []
        assert api.list_repo_tree.call_args.kwargs["path_in_repo"] == "cond_seed42"

    @pytest.mark.parametrize("file_exists,expected_ok", [(True, True), (False, False)])
    def test_exact_file_fallback_marks_present_or_missing(self, tmp_path, file_exists, expected_ok):
        """A cited blob (exact-file) path resolves via the EntryNotFoundError ->
        file_exists fallback, present/missing per the probe."""
        url = "https://huggingface.co/datasets/org/data/blob/main/issue1_x/results.json"
        plan = tmp_path / "plan.md"
        plan.write_text(f"data {url}\n")
        api = MagicMock()
        api.list_repo_tree.side_effect = EntryNotFoundError("entry not found")
        api.file_exists.return_value = file_exists
        with patch("huggingface_hub.HfApi", return_value=api):
            ok, missing = verify_artifacts_exist(plan)
        assert ok is expected_ok
        assert missing == ([] if expected_ok else [url])

    # ── #1482: glob-skip + repo_type-fallback classification arms ─────────────

    def test_glob_path_skipped_never_probed(self, tmp_path, caplog):
        """Acceptance 1 + 6: a glob-bearing hf:// path (the plan's own planned
        output) is SKIPPED — never probed, never in ``missing`` — with one
        INFO log line naming the URL."""
        url = (
            "hf://superkaiba1/explore-persona-space-data/issue1482_error_analysis/"
            "analysis_tensors/early_layer/pooled_l3_*.npz"
        )
        plan = tmp_path / "plan.md"
        plan.write_text(f"planned output at {url}\n")
        api = MagicMock()
        with (
            patch("huggingface_hub.HfApi", return_value=api),
            caplog.at_level(logging.INFO, logger="explore_persona_space.orchestrate.hub"),
        ):
            ok, missing = verify_artifacts_exist(plan)
        assert ok is True
        assert missing == []
        # The probe is NEVER attempted for a glob-shaped path.
        api.repo_info.assert_not_called()
        api.list_repo_tree.assert_not_called()
        api.file_exists.assert_not_called()
        # Observability (acceptance 6): one INFO line per skipped URL.
        skip_lines = [
            r.getMessage()
            for r in caplog.records
            if "glob-shaped hf:// path — planned-output shape, skipped" in r.getMessage()
        ]
        assert skip_lines == [f"glob-shaped hf:// path — planned-output shape, skipped: {url}"]

    def test_unprefixed_uri_retries_as_dataset(self, tmp_path):
        """Acceptance 2: an un-prefixed hf:// citation of a DATASET repo is
        retried under ``repo_type="dataset"`` after the model probe raises
        ``RepositoryNotFoundError`` — and then checks normally."""
        url = "hf://superkaiba1/explore-persona-space-data/issue1482_error_analysis/summary.json"
        plan = tmp_path / "plan.md"
        plan.write_text(f"carry-over input {url}\n")
        api = _RepoTypeFallbackApi(["issue1482_error_analysis/summary.json"])
        with patch("huggingface_hub.HfApi", return_value=api):
            ok, missing = verify_artifacts_exist(plan)
        assert ok is True
        assert missing == []
        assert api.tree_repo_types == ["model", "dataset"]

    def test_neither_repo_type_is_missing_row_path_bearing(self, tmp_path):
        """Acceptance 3, path-bearing variant: a repo resolving under NEITHER
        repo_type yields a ``missing`` row (the URL string), never an
        exception."""
        url = "hf://org/ghost-repo/some/path.json"
        plan = tmp_path / "plan.md"
        plan.write_text(f"reuse {url}\n")
        api = _RepoTypeFallbackApi(dataset_exists=False)
        with patch("huggingface_hub.HfApi", return_value=api):
            ok, missing = verify_artifacts_exist(plan)
        assert ok is False
        assert missing == [url]
        assert api.tree_repo_types == ["model", "dataset"]

    def test_explicit_kind_repo_not_found_is_missing_no_retry(self, tmp_path):
        """An explicit ``datasets/``-prefixed citation whose repo raises
        ``RepositoryNotFoundError`` is a ``missing`` row with NO cross-repo_type
        retry (the kind was explicit)."""
        url = "https://huggingface.co/datasets/org/ghost/tree/main/x"
        plan = tmp_path / "plan.md"
        plan.write_text(f"data {url}\n")
        api = _RepoTypeFallbackApi(dataset_exists=False)
        with patch("huggingface_hub.HfApi", return_value=api):
            ok, missing = verify_artifacts_exist(plan)
        assert ok is False
        assert missing == [url]
        assert api.tree_repo_types == ["dataset"]

    def test_gated_repo_classifies_as_missing(self, tmp_path):
        """The deliberate narrowing: ``GatedRepoError`` subclasses
        ``RepositoryNotFoundError`` on hub 0.36.2, so a gated repo classifies
        as ``missing`` here (Step 6a's auth_check gate owns gated repos and
        runs before 6a.5)."""
        url = "https://huggingface.co/datasets/org/gated/tree/main/x"
        plan = tmp_path / "plan.md"
        plan.write_text(f"data {url}\n")
        api = MagicMock()
        api.list_repo_tree.side_effect = GatedRepoError("gated repo")
        with patch("huggingface_hub.HfApi", return_value=api):
            ok, missing = verify_artifacts_exist(plan)
        assert ok is False
        assert missing == [url]

    def test_transport_auth_error_still_propagates(self, tmp_path):
        """Acceptance 5: a genuine transport/auth error (a 403 that is NOT
        repo-not-found) still PROPAGATES — the fallback narrows only
        ``RepositoryNotFoundError``."""
        plan = tmp_path / "plan.md"
        plan.write_text("adapter https://huggingface.co/org/models/tree/main/cond_seed42\n")
        api = MagicMock()
        api.list_repo_tree.side_effect = _http_err(403, "403 forbidden (auth)")
        with (
            patch("huggingface_hub.HfApi", return_value=api),
            pytest.raises(HfHubHTTPError),
        ):
            verify_artifacts_exist(plan)


class TestHfUrlPathHasGlob:
    """Pin the shared #1482 glob definition both consumers use
    (``verify_artifacts_exist``'s skip arm + ``check_claimed_urls_resolve``'s
    skipped-count disclosure)."""

    @pytest.mark.parametrize("path_piece", ["pooled_l3_*.npz", "shard_?.npz", "shard_[01].npz"])
    def test_glob_metachars_detected(self, path_piece):
        assert hf_url_path_has_glob(f"hf://org/repo/dir/{path_piece}") is True

    def test_web_form_glob_detected(self):
        assert hf_url_path_has_glob("https://huggingface.co/org/repo/tree/main/dir/x_*.json") is (
            True
        )

    def test_plain_path_is_not_glob(self):
        assert hf_url_path_has_glob("hf://org/repo/dir/file.npz") is False

    def test_repo_root_url_is_not_glob(self):
        assert hf_url_path_has_glob("hf://org/repo") is False

    def test_non_hf_url_is_not_glob(self):
        assert hf_url_path_has_glob("https://wandb.ai/t/p/runs/r123") is False


class TestHfUrlRegexCaptures:
    """Pin the _HF_URL_RE capture groups directly against trailing punctuation."""

    def test_json_context_captures_are_clean(self):
        m = _HF_URL_RE.search('"https://huggingface.co/org/repo/tree/main/sub/dir",')
        assert m is not None
        assert m.group("webrepo") == "org/repo"
        assert m.group("webrev") == "main"
        assert m.group("webpath") == "/sub/dir"

    def test_each_guarded_punct_terminates_web_path(self):
        for punct in "\"'`,;}>\\":
            text = f"https://huggingface.co/org/repo/tree/main/file.json{punct} tail"
            m = _HF_URL_RE.search(text)
            assert m is not None, punct
            assert m.group("webrev") == "main", punct
            assert m.group("webpath") == "/file.json", punct

    def test_each_guarded_punct_terminates_uri_revision_and_path(self):
        for punct in "\"'`,;}>\\":
            rev_m = _HF_URL_RE.search(f"hf://org/repo@v1{punct} tail")
            assert rev_m is not None, punct
            assert rev_m.group("urirev") == "v1", punct
            path_m = _HF_URL_RE.search(f"hf://org/repo@v1/ckpt.safetensors{punct} tail")
            assert path_m is not None, punct
            assert path_m.group("uripath") == "/ckpt.safetensors", punct


# ── detect_refusal / filter_refusals ─────────────────────────────────────────


def _judge_client(verdict_json: str) -> MagicMock:
    """Build a mock anthropic client returning a single canned text block."""
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=verdict_json)]
    )
    return client


class TestDetectRefusal:
    def test_true_on_refusal(self):
        client = _judge_client('{"refusal": true, "reasoning": "declined"}')
        assert detect_refusal("I'm sorry, I can't help with that.", client=client) is True
        client.messages.create.assert_called_once()

    def test_false_on_normal_completion(self):
        client = _judge_client('{"refusal": false, "reasoning": "answered"}')
        assert detect_refusal("Fairness means treating people equitably.", client=client) is False

    def test_parses_json_embedded_in_prose(self):
        client = _judge_client('Here is my verdict: {"refusal": true} done.')
        assert detect_refusal("No.", client=client) is True

    def test_raises_on_unparseable_verdict(self):
        client = _judge_client("the model rambled with no json")
        with pytest.raises(ValueError, match="could not parse"):
            detect_refusal("anything", client=client)


class TestFilterRefusals:
    def test_filters_and_counts(self):
        # Judge: index-0 refusal, index-1 keep, index-2 refusal.
        verdicts = iter(
            [
                '{"refusal": true}',
                '{"refusal": false}',
                '{"refusal": true}',
            ]
        )
        client = MagicMock()
        client.messages.create.side_effect = lambda **kw: SimpleNamespace(
            content=[SimpleNamespace(type="text", text=next(verdicts))]
        )

        items = [{"c": "I cannot."}, {"c": "Here's how."}, {"c": "I won't."}]
        kept, skipped = filter_refusals(items, key=lambda r: r["c"], client=client)

        assert skipped == 2
        assert kept == [{"c": "Here's how."}]

    def test_identity_key_on_strings(self):
        verdicts = iter(['{"refusal": false}', '{"refusal": true}'])
        client = MagicMock()
        client.messages.create.side_effect = lambda **kw: SimpleNamespace(
            content=[SimpleNamespace(type="text", text=next(verdicts))]
        )

        kept, skipped = filter_refusals(["good answer", "I refuse"], client=client)
        assert kept == ["good answer"]
        assert skipped == 1
