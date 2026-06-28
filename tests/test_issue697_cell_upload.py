"""Tests for issue #697 plan v5 §4.2 — HF upload resilience.

These pin the EXTERNAL behavior of ``scripts/issue697_cell.py::_upload_cell_artifacts``
after the v3→v5 upload rewrite:

  - the per-cell verify is a per-EXPECTED-file ``HfApi.file_exists`` HEAD, NOT a
    paginated ``list_repo_files`` of the ~64K-file dataset repo;
  - ``create_commit`` + ``file_exists`` ride ``_hf_retry`` exponential backoff on
    transient 5xx (504) — a 504 twice then success returns without raising;
  - a non-transient ``HfHubHTTPError`` (403 auth) raises IMMEDIATELY (no retry);
  - a ``file_exists`` False for one expected path raises ``RuntimeError`` naming it.

All HF I/O is mocked (``huggingface_hub.HfApi`` / ``huggingface_hub.list_repo_files``);
artifact paths use ``tmp_path``. No GPU, no network. Mirrors
``tests/test_issue664_per_cell_upload.py``. ``time.sleep`` is monkeypatched to a
no-op so the backoff does not slow the suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import huggingface_hub  # noqa: E402
import issue697_cell as Cell  # noqa: E402
from huggingface_hub.utils import HfHubHTTPError  # noqa: E402


def _http_error(status: int) -> HfHubHTTPError:
    """An ``HfHubHTTPError`` whose ``.response.status_code`` is ``status``."""
    resp = type("Resp", (), {"status_code": status})()
    err = HfHubHTTPError(f"HTTP {status}")
    err.response = resp
    return err


def _artifacts(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    """One .pt + one _E_metadata.json (tensors) + two raw_completions files."""
    t1 = tmp_path / "marker_sp_swe_seed42.pt"
    t1.write_bytes(b"\x00")
    t2 = tmp_path / "marker_sp_swe_seed42_E_metadata.json"
    t2.write_text("{}")
    r1 = tmp_path / "marker_sp_swe_seed42_p_up_seed42.json"
    r1.write_text("[]")
    return [t1, t2], [r1]


# Module-level scripting state (reset per test by the fixture) — sidesteps the
# RUF012 mutable-class-attribute lint + the `from __future__ import annotations`
# ClassVar import-strip race (the same pattern as tests/test_issue664_per_cell_upload.py).
_COMMIT_RAISES: list[Exception] = []
_FILE_EXISTS_MAP: dict[str, bool] = {}
_COMMIT_CALLS = [0]


class _RecordingHfApi:
    """A fake ``HfApi`` whose ``create_commit`` can be scripted to raise N times
    then succeed (``_COMMIT_RAISES``), and whose ``file_exists`` returns from a
    per-path map (``_FILE_EXISTS_MAP``)."""

    def __init__(self, *a, **k):
        pass

    def create_commit(self, **kwargs):
        i = _COMMIT_CALLS[0]
        _COMMIT_CALLS[0] += 1
        if i < len(_COMMIT_RAISES):
            raise _COMMIT_RAISES[i]
        return type("CommitInfo", (), {"oid": "deadbeef"})()

    def file_exists(self, repo_id, filename, repo_type="dataset"):
        return _FILE_EXISTS_MAP.get(filename, True)


@pytest.fixture(autouse=True)
def _patch_hf(monkeypatch):
    _COMMIT_RAISES.clear()
    _FILE_EXISTS_MAP.clear()
    _COMMIT_CALLS[0] = 0
    monkeypatch.setattr(huggingface_hub, "HfApi", _RecordingHfApi)
    # The function imports HfApi/CommitOperationAdd from huggingface_hub at call
    # time, so patching the package attribute is enough.
    # No-op the backoff sleep so retries don't slow the suite (_hf_retry does
    # `import time; time.sleep(...)`, so patching the module attribute covers it).
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    yield


def test_list_repo_files_never_called(tmp_path, monkeypatch):
    """The verify is per-file file_exists, NEVER a whole-repo list_repo_files."""
    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("list_repo_files must NOT be called (plan §4.2)")

    monkeypatch.setattr(huggingface_hub, "list_repo_files", _boom)
    tensors, raws = _artifacts(tmp_path)
    Cell._upload_cell_artifacts(tensors, raws)
    assert called["n"] == 0


def test_504_retries_then_succeeds(tmp_path):
    """A 504 on create_commit twice then success returns without raising."""
    _COMMIT_RAISES[:] = [_http_error(504), _http_error(504)]
    tensors, raws = _artifacts(tmp_path)
    # file_exists True for everything -> verify passes; the 2 transient 504s retry.
    Cell._upload_cell_artifacts(tensors, raws)
    assert _COMMIT_CALLS[0] == 3  # 2 failures + 1 success


def test_403_raises_immediately_no_retry(tmp_path):
    """A non-transient 403 (auth) raises IMMEDIATELY — no retry loop."""
    _COMMIT_RAISES[:] = [_http_error(403)]
    tensors, raws = _artifacts(tmp_path)
    with pytest.raises(HfHubHTTPError):
        Cell._upload_cell_artifacts(tensors, raws)
    # Only the single attempt fired (403 is not in the transient set).
    assert _COMMIT_CALLS[0] == 1


def test_missing_file_raises_runtimeerror_naming_it(tmp_path):
    """file_exists False for one expected path -> RuntimeError naming the path."""
    tensors, raws = _artifacts(tmp_path)
    missing_repo_path = f"{Cell.HF_TENSOR_PREFIX}/{tensors[0].name}"
    _FILE_EXISTS_MAP[missing_repo_path] = False
    with pytest.raises(RuntimeError, match="verification FAILED"):
        Cell._upload_cell_artifacts(tensors, raws)


def test_hf_retry_exhausts_then_raises_runtimeerror(tmp_path):
    """5 consecutive transient 504s exhaust the budget -> RuntimeError naming what."""
    _COMMIT_RAISES[:] = [_http_error(504)] * 6
    tensors, raws = _artifacts(tmp_path)
    with pytest.raises(RuntimeError, match="create_commit"):
        Cell._upload_cell_artifacts(tensors, raws)
    assert _COMMIT_CALLS[0] == 5  # the default attempts budget
