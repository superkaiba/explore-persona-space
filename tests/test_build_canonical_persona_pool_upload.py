"""#988 site 6: ``build_canonical_persona_pool.upload_file_to_hf``'s post-upload
verify is a single ``HfApi.file_exists`` HEAD probe — never a repo listing
(#920: a full listing of the ~1M-file data repo wedges >600 s) — and a False
probe stays fail-loud (RuntimeError). All network faked at the HfApi boundary
(signature-mirrored stub, per the #906 body-test discipline)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_canonical_persona_pool.py"
_spec = importlib.util.spec_from_file_location("bcpp_issue988", _SCRIPT)
bcpp = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["bcpp_issue988"] = bcpp
_spec.loader.exec_module(bcpp)  # type: ignore[union-attr]


class _FakeApi:
    """Signature-mirroring HfApi stand-in (create_repo/upload_file/file_exists)."""

    def __init__(self, *, exists: bool):
        self._exists = exists
        self.uploads: list[str] = []
        self.file_exists_calls: list[tuple] = []

    def create_repo(self, repo_id, *, repo_type=None, private=False, exist_ok=True):
        return None

    def upload_file(self, *, path_or_fileobj, repo_id, path_in_repo, repo_type):
        self.uploads.append(path_in_repo)

    def file_exists(self, repo_id, filename, *, repo_type=None, revision=None):
        self.file_exists_calls.append((repo_id, filename, repo_type))
        return self._exists

    def list_repo_files(self, *a, **k):  # pragma: no cover - must never run
        raise AssertionError("repo listing must never be called (#920/#988)")


def test_upload_file_to_hf_verifies_via_single_head_probe(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "t")
    fake = _FakeApi(exists=True)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: fake)
    f = tmp_path / "pool.json"
    f.write_text("{}")

    digest = bcpp.upload_file_to_hf(f, "pool.json")

    assert digest == bcpp.sha256_file(f)
    full_path = f"{bcpp.HF_PREFIX}/pool.json"
    assert fake.uploads == [full_path]
    assert fake.file_exists_calls == [(bcpp.HF_DATA_REPO, full_path, "dataset")]


def test_upload_file_to_hf_false_probe_raises(monkeypatch, tmp_path):
    """False -> RuntimeError: the fail-loud contract is pinned (a missing
    upload here means the artifact is lost at pod termination)."""
    monkeypatch.setenv("HF_TOKEN", "t")
    fake = _FakeApi(exists=False)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: fake)
    f = tmp_path / "pool.json"
    f.write_text("{}")

    with pytest.raises(RuntimeError, match="post-upload verify FAIL"):
        bcpp.upload_file_to_hf(f, "pool.json")


def test_no_bare_list_repo_files_import_remains():
    """The module no longer IMPORTS huggingface_hub.list_repo_files (#988) —
    checked via the AST so a prose mention in a comment cannot false-fail."""
    import ast

    tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = {a.name for a in node.names}
            assert "list_repo_files" not in names, "bare list_repo_files import re-introduced"
        if isinstance(node, ast.Import):
            assert not any(a.name == "list_repo_files" for a in node.names)
