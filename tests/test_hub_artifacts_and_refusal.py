"""Tests for the data-eval recurring-fix workstream.

Covers (all network mocked):
  - ``list_repo_files_complete`` paginated enumeration handles a >page-size repo
    (mock yields >7901 entries; assert all are enumerated, folders dropped).
  - ``verify_artifacts_exist`` flags a missing HF/WandB URL and passes when all
    cited artifacts resolve; raises on a malformed plan path.
  - ``detect_refusal`` returns True on a canned refusal and False on a normal
    completion (the Claude judge is mocked).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.hf_api import RepoFile, RepoFolder

from explore_persona_space.eval.refusal import detect_refusal, filter_refusals
from explore_persona_space.orchestrate.hub import (
    _HF_URL_RE,
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


# ── verify_artifacts_exist ────────────────────────────────────────────────────


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
