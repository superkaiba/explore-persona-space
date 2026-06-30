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


_XET_URL = (
    "https://huggingface.co/api/datasets/superkaiba1/explore-persona-space-data/"
    "xet-read-token/3b86d79f"
)


def _make_429(url: str = _XET_URL, retry_after: str | None = None) -> HfHubHTTPError:
    resp = requests.Response()
    resp.status_code = 429
    resp.url = url
    if retry_after is not None:
        resp.headers["Retry-After"] = retry_after
    return HfHubHTTPError(
        f"429 Client Error: Too Many Requests for url: {url}. "
        f"We had to rate limit you, you hit the quota of 2500 api requests per 5 minutes period.",
        response=resp,
    )


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


# ── round-4 crash-fix (#658 failure v10): HF Hub 429 rate-limit retry + throttle ──


@pytest.fixture
def _capture_sleeps(monkeypatch):
    """Record the backoff durations the retry wrapper requests (sleep is no-op)."""
    import time

    seen: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: seen.append(s))
    return seen


def test_429_rate_limit_is_classified():
    mod = _load_module()
    err = _make_429()
    assert mod._is_rate_limit_429(err) is True
    assert mod._is_transient_upload_error(err) is True, "429 must be retried"
    assert mod._is_storage_quota_403(err) is False
    # A 504 is transient but NOT a rate-limit (it takes the short backoff).
    assert mod._is_rate_limit_429(_make_504()) is False


def test_429_is_retried_then_succeeds(_capture_sleeps):
    """Pre-fix the snapshot_download 429 was unwrapped and fatal; this asserts the
    wrapper retries it (the round-4 fix) and eventually returns the result."""
    mod = _load_module()
    calls = {"n": 0}

    def fake_snapshot_download(repo_id, *, repo_type=None, **_kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:  # two rate-limit 429s, then success
            raise _make_429()
        return "/local/pv_store"

    out = mod._retry_on_transient_hf(
        fake_snapshot_download, "repo", repo_type="dataset", _what="snapshot_download:pv_store"
    )
    assert calls["n"] == 3, "should retry twice on 429 then succeed"
    assert out == "/local/pv_store"


def test_429_backoff_is_longer_than_5xx(_capture_sleeps):
    """The 429 backoff (rate-limit window) must be LONGER than the 5xx backoff so
    it doesn't re-trip the 5-min quota by retrying too soon."""
    mod = _load_module()

    # 429 attempt-1 backoff (exponential, no Retry-After header) is >= 60s.
    assert mod._backoff_seconds(_make_429(), 1) >= 60.0
    # 5xx attempt-1 backoff is the original short 10s.
    assert mod._backoff_seconds(_make_504(), 1) == 10.0
    assert mod._backoff_seconds(_make_429(), 1) > mod._backoff_seconds(_make_504(), 1)
    # The 429 backoff caps at the 300s (5-minute) rate-limit window.
    assert mod._backoff_seconds(_make_429(), 8) == 300.0


def test_429_honors_retry_after_header():
    """When HF Hub sets Retry-After, the wrapper sleeps that long (clamped to the
    [60, 300]s window)."""
    mod = _load_module()
    assert mod._backoff_seconds(_make_429(retry_after="120"), 1) == 120.0
    # Below the floor → clamped up to 60s; above the window → clamped to 300s.
    assert mod._backoff_seconds(_make_429(retry_after="5"), 1) == 60.0
    assert mod._backoff_seconds(_make_429(retry_after="9999"), 1) == 300.0
    # Unparseable header → falls back to exponential (>= 60 on attempt 1).
    assert mod._backoff_seconds(_make_429(retry_after="soon"), 1) >= 60.0


def test_429_persistent_reraises_after_max_attempts(_capture_sleeps):
    mod = _load_module()
    calls = {"n": 0}

    def fake(*_a, **_k):
        calls["n"] += 1
        raise _make_429()

    with pytest.raises(HfHubHTTPError):
        mod._retry_on_transient_hf(fake, "r", max_attempts=3, _what="snapshot_download:pv_store")
    assert calls["n"] == 3, "persistent 429 should attempt exactly max_attempts before re-raising"


def test_snapshot_download_called_with_max_workers_4(monkeypatch):
    """The fit's _resolve_pv_store must throttle snapshot_download to max_workers=4
    (defense-in-depth keeping the per-file xet-read-token burst under the 2500/5min
    quota) AND route the call through the 429-aware retry wrapper."""
    import importlib.util
    import sys

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    fit_path = scripts_dir / "issue658_rb_pv_fit.py"
    spec = importlib.util.spec_from_file_location("issue658_rb_pv_fit", fit_path)
    fit = importlib.util.module_from_spec(spec)
    sys.modules["issue658_rb_pv_fit"] = fit
    spec.loader.exec_module(fit)

    captured: dict = {}

    def fake_snapshot_download(repo_id, **kwargs):
        captured["repo_id"] = repo_id
        captured.update(kwargs)
        return "/tmp/fake_local"

    # _resolve_pv_store does `from huggingface_hub import snapshot_download` at
    # call time, so patch it on the huggingface_hub module.
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    class _Args:
        pv_store_dir = None
        pv_store_rev = "main"

    out = fit._resolve_pv_store(_Args(), Path("/tmp/out"))
    assert captured.get("max_workers") == 4, "snapshot_download must be throttled to max_workers=4"
    assert "allow_patterns" in captured, "narrowed-glob allow_patterns must still be threaded"
    assert out == Path("/tmp/fake_local") / f"{fit.HF_PREFIX}/{fit.PV_HF_SUBDIR}"
