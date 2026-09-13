"""Exercise complete upload verification across changed, unchanged and missing files."""

import hashlib
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/workspace_jr_delta_persist.py")
)
PREFIX = "exploratory_workspace_jr/20260912/software_delta_unit_test"


def remote(path, data):
    return RepoFile(
        path=path,
        size=len(data),
        oid=hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest(),
    )


def fixture_api(root, initial, *, mutate=None):
    api = create_autospec(HfApi, instance=True)
    store = dict(initial)
    api.repo_info.return_value = SimpleNamespace(sha="base")

    def get_paths(repo_id, paths, *, expand=False, revision=None, repo_type=None, token=None):
        return [remote(name, store[name]) for name in paths if name in store]

    def upload(
        *,
        repo_id,
        folder_path,
        path_in_repo=None,
        commit_message=None,
        commit_description=None,
        token=None,
        repo_type=None,
        revision=None,
        create_pr=None,
        parent_commit=None,
        allow_patterns=None,
        ignore_patterns=None,
        delete_patterns=None,
        run_as_future=False,
    ):
        assert delete_patterns is None
        for relative in allow_patterns:
            store[f"{path_in_repo}/{relative}"] = (root / relative).read_bytes()
        if mutate:
            mutate(store)
        return SimpleNamespace(oid="new")

    api.get_paths_info.side_effect = get_paths
    api.upload_folder.side_effect = upload
    return api


def test_only_changed_files_upload_but_receipt_covers_every_file(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.txt").write_text("same")
    (root / "b.txt").write_text("changed")
    api = fixture_api(root, {PREFIX + "/a.txt": b"same", PREFIX + "/b.txt": b"old"})
    receipt = tmp_path / "receipt.json"
    result = MODULE["persist"](root, PREFIX, receipt, api)
    assert result["files_uploaded"] == 1 and result["files_verified"] == 2
    assert api.upload_folder.call_args.kwargs["allow_patterns"] == ["b.txt"]
    assert result["revision"] == "new" and receipt.is_file()


def test_noop_still_verifies_full_tree_at_pinned_base(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.txt").write_text("same")
    api = fixture_api(root, {PREFIX + "/a.txt": b"same"})
    result = MODULE["persist"](root, PREFIX, tmp_path / "receipt.json", api)
    assert result["files_uploaded"] == 0 and result["revision"] == "base"
    assert api.get_paths_info.call_count == 2
    api.upload_folder.assert_not_called()


@pytest.mark.parametrize("failure", ["missing", "changed_remote", "changed_local"])
def test_no_receipt_when_final_evidence_differs(tmp_path, failure):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.txt").write_text("same")

    def mutate(store):
        if failure == "missing":
            del store[PREFIX + "/a.txt"]
        elif failure == "changed_remote":
            store[PREFIX + "/a.txt"] = b"bad!"
        else:
            (root / "a.txt").write_text("bad!")

    api = fixture_api(root, {}, mutate=mutate)
    receipt = tmp_path / "receipt.json"
    with pytest.raises(ValueError):
        MODULE["persist"](root, PREFIX, receipt, api)
    assert not receipt.exists()
