"""Regression tests for issue #778 followup reused-artifact sha256 pin (CPU, offline).

Pins the reconciler round-1 CONCERN ``reused-artifact-sha256-pin-dropped``: plan
§12(f) records an ``EXPECTED_SHA256`` content-identity table for the three reused
``r_B`` tensors, and the staging driver MUST fail-loud when a downloaded file's
sha256 does not match — a silently re-uploaded / wrong-generation mirror of the
SAME shape would otherwise pass the shape assert. These tests exercise the pin
logic on temp files; they never touch the Hub.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue778_stage_reused as stage


def test_expected_sha256_table_matches_plan_12f():
    """The three plan §12(f) content-identity pins are present verbatim."""
    assert stage.EXPECTED_SHA256 == {
        "evil": "67d1caafe536f11de29367b48a59f3c6bd372d01a6c44f46a82c6203b1c5ebdb",
        "hallucination": "8bea89cd0e2f43eb902d0fcff544a3eed2fc4006ec79b3bd440b785852db4a6f",
        "sycophancy": "20e498a2a3aca5450c731ac031cc13d887080a432b355e84055bc664d6087ec5",
    }


def test_sha256_file_streams_correct_digest(tmp_path):
    """``_sha256_file`` returns the standard hex sha256 of the file bytes."""
    p = tmp_path / "blob.pt"
    p.write_bytes(b"persona-vector-r_B-bytes")
    assert stage._sha256_file(p) == hashlib.sha256(b"persona-vector-r_B-bytes").hexdigest()


def test_assert_rb_sha256_raises_on_wrong_hash(tmp_path):
    """A file whose sha256 != the recorded value fails loud (wrong-generation mirror)."""
    p = tmp_path / "evil.pt"
    p.write_bytes(b"WRONG generation of the evil r_B tensor")  # not the pinned bytes
    with pytest.raises(RuntimeError, match="content-identity pin"):
        stage.assert_rb_sha256("evil", p)


def test_assert_rb_sha256_passes_on_matching_hash(tmp_path, monkeypatch):
    """A file whose sha256 matches the (monkeypatched) pin is accepted + returns the digest."""
    p = tmp_path / "evil.pt"
    p.write_bytes(b"the-one-true-evil-r_B")
    good = hashlib.sha256(b"the-one-true-evil-r_B").hexdigest()
    monkeypatch.setitem(stage.EXPECTED_SHA256, "evil", good)
    assert stage.assert_rb_sha256("evil", p) == good


def test_assert_rb_sha256_raises_on_unpinned_trait(tmp_path):
    """A trait with no recorded pin is refused (never stage an unpinned reused artifact)."""
    p = tmp_path / "mystery.pt"
    p.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="no EXPECTED_SHA256 pin"):
        stage.assert_rb_sha256("mystery_trait", p)
