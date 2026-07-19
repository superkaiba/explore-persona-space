"""#1547 — bare live HF call sites routed through ``hub.retry_transient``.

Executes the REAL routed function bodies (leg A of #1547) with fakes ONLY at
the external HuggingFace network boundary, per routed site family:

* transient case — a synthetic response-bearing 429 ``HfHubHTTPError`` on the
  first underlying call, success on the second → exactly 2 underlying calls
  and a successful return (acceptance criterion 3);
* non-transient case — a 403-shaped error → exactly 1 underlying call and an
  immediate raise (acceptance criterion 2: behavior preservation — the
  ``_is_transient_upload_error`` gate never retries deterministic failures).

Covered function bodies: ``verify_reused_artifact_keys.resolve_artifact``
(download), ``build_paper.upload_pdf`` (upload), ``sync_models.cmd_pull``
(download loop), ``gen_data_appendix._hf_path`` (download). The remaining
routed scripts (``build_canonical_persona_pool``,
``project_categories_instruct`` / ``_onto_axis``) share the identical 2-line
wrap shape and are pinned statically by
``workflow_lint --check-live-hf-retry-routing``
(tests/test_workflow_lint.py).

Monkeypatch targets are PER-NAME at ``huggingface_hub.<name>`` — the routed
scripts import ``hf_hub_download`` / ``HfApi`` function-locally (the
test_hub_staging_retry.py idiom), so patching the script module would be a
no-op. Until this branch merges, run with ``PYTHONPATH=<worktree>/src`` so
the worktree's ``explore_persona_space`` shadows the editable install.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from explore_persona_space.orchestrate import hub

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _load_script(name: str):
    """Import ``scripts/<name>.py`` as a module (the test_workflow_lint idiom)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _http_err(status: int, msg: str):
    """A response-bearing HfHubHTTPError (the test_issue928_upload_retry idiom)."""
    import requests
    from huggingface_hub.errors import HfHubHTTPError

    resp = requests.Response()
    resp.status_code = status
    return HfHubHTTPError(msg, response=resp)


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """No real sleeps + attempt-bound retry (budget 0 => 6 calls max, #735)."""
    monkeypatch.setattr(hub.time, "sleep", lambda s: None)
    monkeypatch.setenv("EPM_HF_RETRY_BUDGET_S", "0")


def _flaky_download(tmp_path, calls, *, fail_status: int | None = 429, fail_times: int = 1):
    """Signature-conformant ``hf_hub_download`` fake: fail N times, then land."""
    target = tmp_path / "downloaded.bin"
    target.write_text("payload")

    def fake_hf_hub_download(*args, **kwargs):
        calls.append((args, kwargs))
        if fail_status is not None and len(calls) <= fail_times:
            raise _http_err(fail_status, f"{fail_status} synthetic")
        return str(target)

    return fake_hf_hub_download


# ---------------------------------------------------------------------------
# verify_reused_artifact_keys.resolve_artifact — download site
# ---------------------------------------------------------------------------


def test_verify_reused_artifact_keys_download_retries_429_then_succeeds(tmp_path, monkeypatch):
    mod = _load_script("verify_reused_artifact_keys")
    calls: list = []
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _flaky_download(tmp_path, calls))
    args = argparse.Namespace(
        artifact=None, hf_repo="org/repo", hf_path="a/b.pt", repo_type="dataset", revision="main"
    )
    out = mod.resolve_artifact(args)
    assert len(calls) == 2, calls
    assert out == tmp_path / "downloaded.bin"
    assert calls[1][1]["repo_id"] == "org/repo"


def test_verify_reused_artifact_keys_403_raises_first_try(tmp_path, monkeypatch):
    from huggingface_hub.errors import HfHubHTTPError

    mod = _load_script("verify_reused_artifact_keys")
    calls: list = []
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        _flaky_download(tmp_path, calls, fail_status=403, fail_times=99),
    )
    args = argparse.Namespace(
        artifact=None, hf_repo="org/repo", hf_path="a/b.pt", repo_type="dataset", revision=None
    )
    with pytest.raises(HfHubHTTPError, match="403"):
        mod.resolve_artifact(args)
    assert len(calls) == 1, calls


# ---------------------------------------------------------------------------
# build_paper.upload_pdf — upload (commit-API) site
# ---------------------------------------------------------------------------


class _FakeHfApi:
    """Signature-conformant HfApi stand-in for ``upload_pdf`` (upload_file only)."""

    upload_calls: ClassVar[list] = []
    fail_status: int | None = 429
    fail_times: int = 1

    def __init__(self, token=None):
        self.token = token

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        cls = type(self)
        cls.upload_calls.append(path_in_repo)
        if cls.fail_status is not None and len(cls.upload_calls) <= cls.fail_times:
            raise _http_err(cls.fail_status, f"{cls.fail_status} synthetic")
        return SimpleNamespace(oid="deadbeef")


@pytest.fixture()
def fake_hf_api(monkeypatch):
    _FakeHfApi.upload_calls = []
    _FakeHfApi.fail_status = 429
    _FakeHfApi.fail_times = 1
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeHfApi)
    return _FakeHfApi


def test_build_paper_upload_retries_429_then_succeeds(tmp_path, fake_hf_api):
    mod = _load_script("build_paper")
    pdf = tmp_path / "issue_9999.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    url = mod.upload_pdf(pdf, "9999", repo_id="org/data")
    assert len(fake_hf_api.upload_calls) == 2, fake_hf_api.upload_calls
    assert (
        url
        == "https://huggingface.co/datasets/org/data/resolve/deadbeef/papers/issue_9999/issue_9999.pdf"
    )


def test_build_paper_upload_403_raises_first_try(tmp_path, fake_hf_api):
    from huggingface_hub.errors import HfHubHTTPError

    fake_hf_api.fail_status = 403
    fake_hf_api.fail_times = 99
    mod = _load_script("build_paper")
    pdf = tmp_path / "issue_9999.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with pytest.raises(HfHubHTTPError, match="403"):
        mod.upload_pdf(pdf, "9999", repo_id="org/data")
    assert len(fake_hf_api.upload_calls) == 1, fake_hf_api.upload_calls


# ---------------------------------------------------------------------------
# sync_models.cmd_pull — download loop site
# ---------------------------------------------------------------------------


def _pull_args(tmp_path):
    return SimpleNamespace(pull="models/foo", dest=str(tmp_path / "dest"), repo="org/models")


def test_sync_models_pull_loop_retries_429_then_succeeds(tmp_path, monkeypatch, capsys):
    mod = _load_script("sync_models")
    calls: list = []
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _flaky_download(tmp_path, calls))
    monkeypatch.setattr(
        hub, "list_repo_files_complete", lambda api, repo_id, repo_type: ["models/foo/a.bin"]
    )
    mod.cmd_pull(_pull_args(tmp_path))
    assert len(calls) == 2, calls
    assert calls[1][1]["filename"] == "models/foo/a.bin"
    assert "Download complete" in capsys.readouterr().out


def test_sync_models_pull_loop_403_exits_nonzero_first_try(tmp_path, monkeypatch):
    mod = _load_script("sync_models")
    calls: list = []
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        _flaky_download(tmp_path, calls, fail_status=403, fail_times=99),
    )
    monkeypatch.setattr(
        hub, "list_repo_files_complete", lambda api, repo_id, repo_type: ["models/foo/a.bin"]
    )
    # cmd_pull's own catch-all converts the first-try raise to sys.exit(1) —
    # the pre-#1547 behavior for a deterministic failure, preserved.
    with pytest.raises(SystemExit) as exc:
        mod.cmd_pull(_pull_args(tmp_path))
    assert exc.value.code == 1
    assert len(calls) == 1, calls


# ---------------------------------------------------------------------------
# gen_data_appendix._hf_path — download site
# ---------------------------------------------------------------------------


def test_gen_data_appendix_hf_path_retries_429_then_succeeds(tmp_path, monkeypatch):
    mod = _load_script("gen_data_appendix")
    calls: list = []
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _flaky_download(tmp_path, calls))
    out = mod._hf_path("issue1/x.json")
    assert len(calls) == 2, calls
    assert out == str(tmp_path / "downloaded.bin")


def test_gen_data_appendix_hf_path_403_raises_first_try(tmp_path, monkeypatch):
    from huggingface_hub.errors import HfHubHTTPError

    mod = _load_script("gen_data_appendix")
    calls: list = []
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        _flaky_download(tmp_path, calls, fail_status=403, fail_times=99),
    )
    with pytest.raises(HfHubHTTPError, match="403"):
        mod._hf_path("issue1/x.json")
    assert len(calls) == 1, calls
