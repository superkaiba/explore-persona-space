"""Regression: the post-upload HF verify listing must survive a transient 504.

Round-3 crash-fix for #658 follow-up `persona-vectors-style-rb`. Two consecutive
upload runs committed all 12081 files, then crashed rc=1 when the post-upload
``api.list_repo_files`` verify call paginated ``/api/.../tree/main`` and a
follow-up cursor page returned a 504 Gateway Time-out. ``huggingface_hub``'s
pagination retries only 429 on follow-up pages (``http_backoff(...,
retry_on_status_codes=429)``), so the 504 raised ``HfHubHTTPError(504)`` straight
through the (previously unwrapped) ``list_repo_files`` call.

The fix wraps every transient-prone HF call — upload AND the verify listing — in
``_retry_on_transient_hf``, which retries on 5xx/timeout. This test trips the
guard: it asserts the verify-shaped call is retried on a simulated ``tree/main``
504 and eventually succeeds, that the transient classifier recognizes that 504,
that storage-quota-403 still re-raises immediately (overflow fallback intact),
and that a genuinely non-transient error (404) is NOT retried.

Pre-fix (bare ``api.list_repo_files`` with no retry), ``test_tree504_*`` would
fail because the 504 would propagate on the first attempt.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import requests
from huggingface_hub.errors import HfHubHTTPError

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "issue658_extract_rb_personavectors.py"
_TREE_URL = (
    "https://huggingface.co/api/datasets/superkaiba1/explore-persona-space-data/"
    "tree/main?expand=false&recursive=true&limit=1000&cursor=ZZZ"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("i658_extract_rb", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_504(url: str = _TREE_URL) -> HfHubHTTPError:
    resp = requests.Response()
    resp.status_code = 504
    resp.url = url
    return HfHubHTTPError(f"504 Server Error: Gateway Time-out for url: {url}", response=resp)


@pytest.fixture(autouse=True)
def _instant_sleep(monkeypatch):
    """Make retry backoff instant so the test runs fast."""
    import time

    monkeypatch.setattr(time, "sleep", lambda _s: None)


def test_tree504_on_verify_listing_is_retried_then_succeeds():
    mod = _load_module()
    calls = {"n": 0}

    def fake_list_repo_files(repo_id, *, repo_type=None):
        calls["n"] += 1
        if calls["n"] <= 2:  # two transient 504s, then success
            raise _make_504()
        return ["issue658_pvrb/personavectors_rb/rb_extract_manifest.json"]

    out = mod._retry_on_transient_hf(
        fake_list_repo_files, "repo", repo_type="dataset", _what="list_repo_files"
    )
    assert calls["n"] == 3, "should retry twice then succeed"
    assert out == ["issue658_pvrb/personavectors_rb/rb_extract_manifest.json"]


def test_tree504_is_classified_transient():
    mod = _load_module()
    err = _make_504()
    assert mod._is_transient_upload_error(err) is True
    assert mod._is_storage_quota_403(err) is False


def test_storage_quota_403_reraises_immediately():
    mod = _load_module()
    calls = {"n": 0}

    def fake_upload(**_kwargs):
        calls["n"] += 1
        raise Exception("403 Forbidden: You have exceeded your public storage space")

    with pytest.raises(Exception, match="storage"):
        mod._retry_on_transient_hf(fake_upload, _what="upload_folder", folder_path="x")
    assert calls["n"] == 1, "quota-403 must not retry (overflow fallback must still fire)"


def test_non_transient_404_not_retried():
    mod = _load_module()
    calls = {"n": 0}

    def fake(*_a, **_k):
        calls["n"] += 1
        resp = requests.Response()
        resp.status_code = 404
        resp.url = "x"
        raise HfHubHTTPError("404 Not Found", response=resp)

    with pytest.raises(HfHubHTTPError):
        mod._retry_on_transient_hf(fake, "r")
    assert calls["n"] == 1, "404 is non-transient and must re-raise on the first attempt"


def test_persistent_504_reraises_after_max_attempts():
    mod = _load_module()
    calls = {"n": 0}

    def fake(*_a, **_k):
        calls["n"] += 1
        raise _make_504()

    with pytest.raises(HfHubHTTPError):
        mod._retry_on_transient_hf(fake, "r", max_attempts=3, _what="list_repo_files")
    assert calls["n"] == 3, "should attempt exactly max_attempts before re-raising"
