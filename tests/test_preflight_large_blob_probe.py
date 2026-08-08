"""Pins for ``check_hf_large_blob_get`` — the RunPod HF-CDN zero-byte large-blob probe (#2185).

All offline: the opener seam is injected — no network call anywhere in this
suite. Fail-loud pins (plan v3): case 1 (`test_zero_byte_body_warns_and_ok_stays_true`)
pins that a 206 zero-byte body adds a WARNING without raising and that
``report.ok`` stays True (WARN-only — the probe must never be able to flip the
verdict); case 4 (`test_raising_opener_is_swallowed_silently`) pins that an
opener which raises is swallowed to a silent inconclusive (no warning, no
error, no exception escapes past the probe). A future edit that turns the
probe into a hard FAIL, ungates it from RunPod, or lets an exception escape
fails this suite rather than shipping green.
"""

from explore_persona_space.orchestrate import preflight
from explore_persona_space.orchestrate.preflight import (
    DEFAULT_LARGE_BLOB_URL,
    LARGE_BLOB_RANGE_BYTES,
    PreflightReport,
    check_hf_large_blob_get,
)


class _FakeResponse:
    """Minimal stand-in for ``http.client.HTTPResponse`` (context manager + read)."""

    def __init__(self, body: bytes, status: int = 206):
        self._body = body
        self.status = status
        self._pos = 0

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = len(self._body) - self._pos
        chunk = self._body[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _opener_for(body: bytes, status: int = 206, calls: list | None = None):
    """Signature-conformant fake of ``urllib.request.urlopen(req, timeout=...)``."""

    def opener(req, timeout=None):
        if calls is not None:
            calls.append(req)
        return _FakeResponse(body, status=status)

    return opener


def _clean_env(monkeypatch):
    monkeypatch.delenv("EPM_SKIP_LARGE_BLOB_PROBE", raising=False)
    monkeypatch.delenv("EPM_PREFLIGHT_LARGE_BLOB_URL", raising=False)


def test_zero_byte_body_warns_and_ok_stays_true(monkeypatch):
    """Case 1: 206 + zero-byte body -> exactly one WARNING with the diagnosis;
    ``report.ok`` stays True and no error is added (fail-open pinned, not prose)."""
    _clean_env(monkeypatch)
    monkeypatch.setattr(preflight, "is_runpod_env", lambda: True)
    report = PreflightReport()
    check_hf_large_blob_get(report, opener=_opener_for(b"", status=206))
    assert len(report.warnings) == 1, report.warnings
    warning = report.warnings[0]
    assert "HF large-blob GET returned 206 with 0 bytes" in warning
    assert "DNS-steered" in warning
    assert ".claude/rules/gotchas.md" in warning
    assert "parallel-rsync relay" in warning
    # The WARN-only pin: the probe must never call add_error or flip ok.
    assert report.ok is True
    assert report.errors == []


def test_short_read_warns_with_byte_count(monkeypatch):
    """A truncated (short, EOF-terminated) body is the same signature as zero bytes."""
    _clean_env(monkeypatch)
    monkeypatch.setattr(preflight, "is_runpod_env", lambda: True)
    report = PreflightReport()
    check_hf_large_blob_get(report, opener=_opener_for(b"x" * 1024, status=206))
    assert len(report.warnings) == 1, report.warnings
    assert "with 1024 bytes" in report.warnings[0]
    assert report.ok is True
    assert report.errors == []


def test_healthy_full_range_body_is_silent(monkeypatch):
    """Case 2: a full >=1 MiB body -> no warning, no error, ok unchanged."""
    _clean_env(monkeypatch)
    monkeypatch.setattr(preflight, "is_runpod_env", lambda: True)
    report = PreflightReport()
    body = b"x" * LARGE_BLOB_RANGE_BYTES
    check_hf_large_blob_get(report, opener=_opener_for(body, status=206))
    assert report.warnings == []
    assert report.errors == []
    assert report.ok is True


def test_non_runpod_env_does_not_run(monkeypatch):
    """Case 3: off RunPod the probe does not run at all — no opener call, no output."""
    _clean_env(monkeypatch)
    monkeypatch.setattr(preflight, "is_runpod_env", lambda: False)
    calls: list = []
    report = PreflightReport()
    check_hf_large_blob_get(report, opener=_opener_for(b"", calls=calls))
    assert calls == []  # gate pinned: zero network calls off-pod
    assert report.warnings == []
    assert report.errors == []
    assert report.ok is True


def test_raising_opener_is_swallowed_silently(monkeypatch):
    """Case 4: any opener exception (DNS failure, 404 HTTPError, timeout) degrades
    to a silent inconclusive — no warning, no error, no exception escapes."""
    _clean_env(monkeypatch)
    monkeypatch.setattr(preflight, "is_runpod_env", lambda: True)

    def opener(req, timeout=None):
        raise OSError("simulated DNS failure")

    report = PreflightReport()
    check_hf_large_blob_get(report, opener=opener)  # must not raise
    assert report.warnings == []
    assert report.errors == []
    assert report.ok is True


def test_kill_switch_skips_probe(monkeypatch):
    """EPM_SKIP_LARGE_BLOB_PROBE=1 skips the probe even on RunPod."""
    _clean_env(monkeypatch)
    monkeypatch.setattr(preflight, "is_runpod_env", lambda: True)
    monkeypatch.setenv("EPM_SKIP_LARGE_BLOB_PROBE", "1")
    calls: list = []
    report = PreflightReport()
    check_hf_large_blob_get(report, opener=_opener_for(b"", calls=calls))
    assert calls == []
    assert report.warnings == []
    assert report.ok is True


def test_url_override_and_range_header(monkeypatch):
    """EPM_PREFLIGHT_LARGE_BLOB_URL overrides the default target, and the request
    carries the 1 MiB Range header (verdict keys on bytes, not throughput)."""
    _clean_env(monkeypatch)
    monkeypatch.setattr(preflight, "is_runpod_env", lambda: True)
    custom = "https://huggingface.co/some/other/resolve/main/blob.bin"
    monkeypatch.setenv("EPM_PREFLIGHT_LARGE_BLOB_URL", custom)
    calls: list = []
    report = PreflightReport()
    check_hf_large_blob_get(report, opener=_opener_for(b"x" * LARGE_BLOB_RANGE_BYTES, calls=calls))
    assert len(calls) == 1
    req = calls[0]
    assert req.full_url == custom
    assert req.get_header("Range") == f"bytes=0-{LARGE_BLOB_RANGE_BYTES - 1}"
    assert custom != DEFAULT_LARGE_BLOB_URL
    assert report.warnings == []
