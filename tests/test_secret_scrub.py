"""Tests for the secret upload gate + scrub tooling (secret_scrub.py).

Pins, in order of blast radius:

1. The gate is WIRED — ``hub._upload`` and
   ``upload_sharded.upload_dir_sharded`` both call
   ``assert_upload_clean`` before any HF API object is constructed
   (string-level pin, same idiom as test_workflow_hub_upload_as_file.py:
   a refactor cannot silently drop the call).
2. Real-secret-grade strings are FOUND (per pattern class), placeholders
   and documentation examples are NOT (the dummy filter — a false
   positive wedges an unattended multi-hour upload run).
3. ``scrub_file`` is same-length, tar-offset-preserving, idempotent, and
   fails loud rather than reporting clean when residue survives.
4. The kill switch (``EPM_SECRET_UPLOAD_GATE=0``) and the binary-skip
   policy behave as documented.

Fixture tokens are ASSEMBLED AT RUNTIME (never stored literally) so this
file cannot itself trip GitHub push protection or HF secret scanning —
the same failure class it exists to prevent (see commit 8c2523dc for the
original scanner-tripping incident).
"""

from __future__ import annotations

import tarfile

import pytest

from explore_persona_space.orchestrate import secret_scrub
from explore_persona_space.orchestrate.secret_scrub import (
    Finding,
    SecretUploadGateError,
    assert_upload_clean,
    scan_bytes,
    scan_file,
    scrub_bytes,
    scrub_file,
)


def _openai_key() -> bytes:
    # sk- + filler + the base64("OpenAI") marker + tail, assembled at runtime.
    return b"sk-" + b"Wq8rBv31Jm" + b"T3BlbkFJ" + b"q9DkLm20ZnRs7Yw4"


def _hf_token() -> bytes:
    return b"hf_" + b"ZyKQmwRbNvTsLdCeGhJa" + b"PoUwYbNvTsQe"


def _github_pat() -> bytes:
    return b"github" + b"_pat_" + b"22AB0CDEFG" + b"hijklmnopqrstuv0123456789"


def _jwt() -> bytes:
    return (
        b"eyJ" + b"hbGciOiJSUzI1NiIsImtpZCI6IjEifQ"
        + b"." + b"eyJ" + b"zdWIiOiIwMHVra2k0OHBzIiwibmFtZSI6IkEifQ"
        + b"." + b"q9DkLm20ZnRs7Yw4Wq8rBv31JmKQmwRb"
    )


# ---------------------------------------------------------------- detection


@pytest.mark.parametrize(
    "name,payload",
    [
        ("openai-real", _openai_key()),
        ("hf-token", _hf_token()),
        ("github-pat-fine", _github_pat()),
        ("jwt-signed", _jwt()),
        ("aws-access-key", b"AKIA" + b"IOSFODNN7EX4MPLQ"),
        # the finding covers the regex match itself (no scheme/host prefix)
        ("infura-url-key", b"infura.io/v3/" + b"9aa3d95b3bc440fa88ea12eaa4456161"),
    ],
)
def test_real_secret_shapes_are_found(name, payload):
    data = b'{"text": "then I ran it with ' + payload + b' and it worked"}'
    hits = scan_bytes(data)
    assert [h.pattern for h in hits] == [name]
    # offsets/lengths point at the exact payload bytes
    h = hits[0]
    assert data[h.offset : h.offset + h.length] == payload


@pytest.mark.parametrize(
    "payload",
    [
        b"sk-" + b"X" * 48,  # placeholder fill
        b"hf_" + b"x" * 8 + b"EXAMPLE" + b"x" * 20,  # example marker in context
        b"https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
        b"sk-YOUR_API_KEY_GOES_HERE_1234567890abcdefghijkl",
    ],
)
def test_placeholders_are_not_findings(payload):
    data = b'{"text": "set it to ' + payload + b' in your config"}'
    assert scan_bytes(data) == []


def test_long_test_identifiers_are_not_findings():
    # The HF "Lob" false-positive class: snake_case pytest names of exactly
    # the fatal length must never trip the gate.
    data = b"def test_cookie_samesite_lax_without_session(self):\n    pass\n"
    assert scan_bytes(data) == []


# ---------------------------------------------------------------- scrubbing


def test_scrub_bytes_same_length_and_clean():
    key = _openai_key()
    data = b'{"a": "' + key + b'", "b": 1}'
    out, findings = scrub_bytes(data)
    assert len(out) == len(data)
    assert len(findings) == 1
    assert key not in out
    assert scan_bytes(out) == []
    # JSON structure intact (X-fill is alphanumeric)
    import json

    assert json.loads(out)["b"] == 1


def test_scrub_file_plain(tmp_path):
    f = tmp_path / "pool.jsonl"
    f.write_bytes(b'{"t": "' + _hf_token() + b'"}\n')
    size = f.stat().st_size
    findings = scrub_file(f)
    assert len(findings) == 1
    assert f.stat().st_size == size
    assert scan_file(f) == []
    assert scrub_file(f) == []  # idempotent


def test_scrub_file_tar_preserves_offsets(tmp_path):
    """Members are patched at their absolute archive offsets: sizes, order,
    and every other member's bytes stay identical — the __packed__ index
    offset contract."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "clean.json").write_bytes(b'{"ok": true}')
    (src / "dirty.json").write_bytes(b'{"k": "' + _openai_key() + b'"}')
    tar_path = tmp_path / "shard.tar"
    with tarfile.open(tar_path, "w") as tf:
        tf.add(src / "clean.json", arcname="p/clean.json")
        tf.add(src / "dirty.json", arcname="p/dirty.json")
    before_size = tar_path.stat().st_size
    with tarfile.open(tar_path) as tf:
        clean_before = tf.extractfile("p/clean.json").read()

    findings = scrub_file(tar_path)
    assert [f.member for f in findings] == ["p/dirty.json"]
    assert tar_path.stat().st_size == before_size
    with tarfile.open(tar_path) as tf:  # still a valid archive
        assert tf.extractfile("p/clean.json").read() == clean_before
        dirty = tf.extractfile("p/dirty.json").read()
    assert _openai_key() not in dirty
    assert len(dirty) == len(b'{"k": "' + _openai_key() + b'"}')
    assert scan_file(tar_path) == []


def test_scrub_file_fails_loud_on_residue(tmp_path, monkeypatch):
    """A fix that does not actually remove the finding must raise, never
    report clean: pin the post-patch rescan by making scan_file always
    return the same finding (as if the patch missed)."""
    f = tmp_path / "x.json"
    f.write_bytes(b'{"t": "' + _hf_token() + b'"}')
    frozen = secret_scrub.scan_file(f)
    assert len(frozen) == 1
    monkeypatch.setattr(secret_scrub, "scan_file", lambda p: frozen)
    with pytest.raises(RuntimeError, match="refusing to report clean"):
        scrub_file(f)


# ---------------------------------------------------------------- the gate


def test_gate_raises_on_text_hit(tmp_path):
    d = tmp_path / "up"
    d.mkdir()
    (d / "a.json").write_bytes(b'{"k": "' + _openai_key() + b'"}')
    with pytest.raises(SecretUploadGateError) as ei:
        assert_upload_clean([d], what="test")
    msg = str(ei.value)
    assert "REFUSING" in msg
    assert "scrub_secrets.py" in msg
    assert _openai_key().decode() not in msg  # values never printed


def test_gate_passes_clean_dir(tmp_path):
    d = tmp_path / "up"
    d.mkdir()
    (d / "a.json").write_bytes(b'{"ok": 1}')
    assert_upload_clean([d], what="test")


def test_gate_kill_switch(tmp_path, monkeypatch):
    d = tmp_path / "up"
    d.mkdir()
    (d / "a.json").write_bytes(b'{"k": "' + _openai_key() + b'"}')
    monkeypatch.setenv("EPM_SECRET_UPLOAD_GATE", "0")
    assert_upload_clean([d], what="test")  # no raise


def test_gate_skips_binary_by_default(tmp_path):
    d = tmp_path / "up"
    d.mkdir()
    (d / "capture.pt").write_bytes(b"blob " + _openai_key() + b" blob")
    assert_upload_clean([d], what="test")  # binary skipped, no raise


def test_gate_scans_tar_members(tmp_path):
    src = tmp_path / "f.json"
    src.write_bytes(b'{"k": "' + _hf_token() + b'"}')
    tar_path = tmp_path / "shard.tar"
    with tarfile.open(tar_path, "w") as tf:
        tf.add(src, arcname="p/f.json")
    with pytest.raises(SecretUploadGateError):
        assert_upload_clean([tar_path], what="test")


# ------------------------------------------------------------- wiring pins


def test_gate_wired_into_hub_upload():
    """hub._upload must call assert_upload_clean before HfApi construction —
    a refactor dropping the call reopens the 2026-08-17 leak path."""
    from pathlib import Path as P

    import explore_persona_space.orchestrate.hub as hub

    src = P(hub.__file__).read_text()
    body = src.split("def _upload(")[1].split("\ndef ")[0]
    assert "assert_upload_clean(" in body
    assert body.index("assert_upload_clean(") < body.index("HfApi(token=token)")


def test_gate_wired_into_upload_dir_sharded():
    from pathlib import Path as P

    import explore_persona_space.orchestrate.upload_sharded as us

    src = P(us.__file__).read_text()
    body = src.split("def upload_dir_sharded(")[1]
    assert "assert_upload_clean(" in body
    # before the headroom probe / any commit machinery
    assert body.index("assert_upload_clean(") < body.index("check_projected_upload_headroom(")


def test_hub_upload_gate_fires_before_network(tmp_path, monkeypatch):
    """End-to-end through hub._upload: a dirty folder raises the gate error
    with no HfApi ever constructed."""
    import huggingface_hub

    import explore_persona_space.orchestrate.hub as hub

    monkeypatch.setenv("HF_TOKEN", "hf_" + "x" * 34)  # gate runs pre-API

    constructed = []

    class _BoomApi:
        def __init__(self, *a, **k):
            constructed.append(1)

    # hub._upload imports HfApi function-locally from huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", _BoomApi)
    d = tmp_path / "up"
    d.mkdir()
    (d / "a.json").write_bytes(b'{"k": "' + _openai_key() + b'"}')
    with pytest.raises(SecretUploadGateError):
        hub._upload(d, "user/repo", "dataset", "prefix")
    assert constructed == []


def test_finding_where_formats():
    f = Finding(path="a.tar", member="m/x.json", offset=5, length=3, pattern="p", masked="a…b")
    assert f.where() == "a.tar::m/x.json @5"
    f2 = Finding(path="x.json", member="", offset=5, length=3, pattern="p", masked="a…b")
    assert f2.where() == "x.json @5"
