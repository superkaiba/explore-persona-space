"""Tests for #2097 — the default-ON staging disk-headroom assert.

Pins (a) the `stage_hub_prefix` prefix-level headroom refusal (REAL
`_assert_stage_headroom` + REAL `assert_out_root_headroom` body — the
low-headroom refusal is a `pytest.raises(RuntimeError)`, never a silent
skip), (b) missing-files-only sizing (an all-staged resume never asserts),
(c) the `EPM_HF_STAGE_HEADROOM_SKIP=1` kill switch logging loud, (d) the
unknown-size degrade (any-None trigger; 1 GB floor), (e) the
`EPM_HF_STAGE_HEADROOM_FACTOR` env resolver's garbled-fallback warning,
(f) `stage_hub_file(size_bytes=...)`'s opt-in single-file assert (default
None = byte-identical legacy path), and (g) the
`list_hf_files_under_path` -> `list_hf_entries_under_path` delegation
(same paths, ONE listing serves both paths and sizes).

Monkeypatch targets are PER-NAME (the test_hub_staging_retry.py
convention): `HfApi` / `hf_hub_download` are function-body lazy imports and
patch at `huggingface_hub.<name>`; `list_hf_entries_under_path`,
`stage_hub_file`, and `_assert_stage_headroom` ARE hub-module globals and
patch at the hub site; the lazily-imported `assert_out_root_headroom`
resolves at call time from the preflight module and patches THERE.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.hf_api import RepoFile

from explore_persona_space.orchestrate import hub, preflight


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """No real sleeps + attempt-bound retry (budget 0 => 6 calls max, #735)."""
    monkeypatch.setattr(hub.time, "sleep", lambda s: None)
    monkeypatch.setenv("EPM_HF_RETRY_BUDGET_S", "0")


class _FakeApi:
    """HfApi stand-in for the stage_hub_prefix seam (repo_info only)."""

    def __init__(self, token=None):
        self.token = token

    def repo_info(self, repo_id, repo_type=None):
        return SimpleNamespace(sha="abc123")


def _fake_stage_recorder(calls: list):
    """Signature-conformant stage_hub_file fake (mirrors the real def,
    incl. the #2097 ``size_bytes`` kwarg stage_hub_prefix must NOT pass)."""

    def fake_stage(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
        size_bytes=None,
    ):
        assert size_bytes is None  # prefix-level assert covers it — never threaded
        calls.append(path_in_repo)
        return Path(target)

    return fake_stage


# ---------------------------------------------------------------------------
# Factor env resolver
# ---------------------------------------------------------------------------


def test_stage_headroom_factor_default_and_env(monkeypatch):
    monkeypatch.delenv("EPM_HF_STAGE_HEADROOM_FACTOR", raising=False)
    assert hub._stage_headroom_factor() == 1.5
    monkeypatch.setenv("EPM_HF_STAGE_HEADROOM_FACTOR", "2.0")
    assert hub._stage_headroom_factor() == 2.0


def test_stage_headroom_factor_garbled_falls_back_with_warning(monkeypatch, caplog):
    """Garbled / non-positive env falls back to 1.5 WITH a logged warning —
    never raises, never silently zeroes the floor."""
    for bad in ("not-a-float", "-1", "0", "nan"):
        monkeypatch.setenv("EPM_HF_STAGE_HEADROOM_FACTOR", bad)
        with caplog.at_level(logging.WARNING, logger=hub.logger.name):
            caplog.clear()
            assert hub._stage_headroom_factor() == 1.5
            assert any("EPM_HF_STAGE_HEADROOM_FACTOR" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# stage_hub_prefix: refusal (REAL assert bodies) + pass + resume + kill switch
# ---------------------------------------------------------------------------


def test_stage_hub_prefix_refuses_on_low_headroom_real_assert_body(tmp_path, monkeypatch):
    """The low-headroom refusal executes the REAL `_assert_stage_headroom`
    AND the REAL `assert_out_root_headroom` body (statvfs + mount
    resolution) against a need (1 PB x 1.5) no test filesystem satisfies:
    RuntimeError naming the mount, raised BEFORE any download — zero
    stage_hub_file calls."""
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)
    monkeypatch.setattr(
        hub, "list_hf_entries_under_path", lambda *a, **k: [("pfx/huge.bin", 10**15)]
    )
    calls: list[str] = []
    monkeypatch.setattr(hub, "stage_hub_file", _fake_stage_recorder(calls))

    with pytest.raises(RuntimeError, match=r"disk-headroom.*hub-staging"):
        hub.stage_hub_prefix("org/data", "pfx", tmp_path / "dest")
    assert calls == []  # refusal fires BEFORE the download pool


def test_stage_hub_prefix_passes_and_stages_when_headroom_ok(tmp_path, monkeypatch):
    """Tiny sizes clear the (real) statvfs floor; the canary is faked at the
    filesystem boundary (signature-conformant `_probe_writable_bytes` stand-in
    — its real body has its own pins in test_preflight_disk.py) so the test
    never fallocates 1 GB. Every file stages exactly once."""
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)
    monkeypatch.setattr(
        hub,
        "list_hf_entries_under_path",
        lambda *a, **k: [("pfx/a.json", 10), ("pfx/sub/b.json", 20)],
    )

    def fake_probe(check_path: str, probe_bytes: int):
        return True, None

    monkeypatch.setattr(preflight, "_probe_writable_bytes", fake_probe)
    calls: list[str] = []
    monkeypatch.setattr(hub, "stage_hub_file", _fake_stage_recorder(calls))

    dest = tmp_path / "dest"
    out = hub.stage_hub_prefix("org/data", "pfx", dest)
    assert sorted(calls) == ["pfx/a.json", "pfx/sub/b.json"]
    assert out == [dest / "pfx/a.json", dest / "pfx/sub/b.json"]


def test_stage_hub_prefix_skip_existing_resume_never_asserts(tmp_path, monkeypatch):
    """All targets already staged => missing is empty => the headroom assert
    is NEVER invoked (spy) and the staged targets return (AC1's resume no-op)."""
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)
    monkeypatch.setattr(
        hub,
        "list_hf_entries_under_path",
        lambda *a, **k: [("pfx/a.json", 10**15), ("pfx/b.json", 10**15)],
    )
    asserts: list = []
    monkeypatch.setattr(hub, "_assert_stage_headroom", lambda *a, **k: asserts.append((a, k)))
    dest = tmp_path / "dest"
    for rel in ("pfx/a.json", "pfx/b.json"):
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"already staged")

    out = hub.stage_hub_prefix("org/data", "pfx", dest)
    assert asserts == []
    assert out == [dest / "pfx/a.json", dest / "pfx/b.json"]


def test_stage_hub_prefix_kill_switch_skips_loud(tmp_path, monkeypatch, caplog):
    """EPM_HF_STAGE_HEADROOM_SKIP=1: no refusal even at absurd need, ONE loud
    warning naming the switch — never a silent skip."""
    monkeypatch.setenv("EPM_HF_STAGE_HEADROOM_SKIP", "1")
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)
    monkeypatch.setattr(
        hub, "list_hf_entries_under_path", lambda *a, **k: [("pfx/huge.bin", 10**15)]
    )
    calls: list[str] = []
    monkeypatch.setattr(hub, "stage_hub_file", _fake_stage_recorder(calls))

    with caplog.at_level(logging.WARNING, logger=hub.logger.name):
        out = hub.stage_hub_prefix("org/data", "pfx", tmp_path / "dest")
    assert calls == ["pfx/huge.bin"]
    assert out == [tmp_path / "dest" / "pfx/huge.bin"]
    assert any("EPM_HF_STAGE_HEADROOM_SKIP" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Sizing: unknown-size degrade + missing-only arithmetic (spy on need_gb)
# ---------------------------------------------------------------------------


def _spy_out_root_assert(monkeypatch) -> list[dict]:
    """Signature-conformant assert_out_root_headroom spy at the preflight
    module (the lazy import resolves there at call time)."""
    seen: list[dict] = []

    def fake_assert(out_root, need_gb, *, phase="", canary_gb=1.0):
        seen.append({"out_root": Path(out_root), "need_gb": need_gb, "phase": phase})
        return 100.0

    monkeypatch.setattr(preflight, "assert_out_root_headroom", fake_assert)
    return seen


def test_assert_stage_headroom_all_unknown_sizes_degrades_to_floor(monkeypatch, tmp_path, caplog):
    """All-None sizes => need_gb = 1 GB floor x 1.5 factor, with the loud
    n_unknown/n_missing degrade warning."""
    seen = _spy_out_root_assert(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=hub.logger.name):
        hub._assert_stage_headroom(tmp_path, [("a", None), ("b", None)], what="org/data@r:pfx")
    assert len(seen) == 1
    assert seen[0]["need_gb"] == pytest.approx(1.5)
    assert seen[0]["phase"] == "hub-staging"
    assert any("2 of 2 missing file(s)" in r.message for r in caplog.records)


def test_assert_stage_headroom_mixed_sizes_warns_and_sums_known(monkeypatch, tmp_path, caplog):
    """MIXED known+None: the degrade warning fires on ANY None (the #541
    partial-None shape) while need_gb sums the KNOWN sizes (3 GB x 1.5)."""
    seen = _spy_out_root_assert(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=hub.logger.name):
        hub._assert_stage_headroom(
            tmp_path, [("a", 3_000_000_000), ("b", None)], what="org/data@r:pfx"
        )
    assert len(seen) == 1
    assert seen[0]["need_gb"] == pytest.approx(4.5)
    assert any("1 of 2 missing file(s)" in r.message for r in caplog.records)


def test_assert_stage_headroom_garbled_factor_uses_default(monkeypatch, tmp_path, caplog):
    """Garbled EPM_HF_STAGE_HEADROOM_FACTOR => default 1.5 + warning, and the
    assert still runs (10 GB known x 1.5 = 15)."""
    monkeypatch.setenv("EPM_HF_STAGE_HEADROOM_FACTOR", "garbled")
    seen = _spy_out_root_assert(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=hub.logger.name):
        hub._assert_stage_headroom(tmp_path, [("a", 10_000_000_000)], what="w")
    assert seen[0]["need_gb"] == pytest.approx(15.0)
    assert any("EPM_HF_STAGE_HEADROOM_FACTOR" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# stage_hub_file: opt-in size_bytes arm
# ---------------------------------------------------------------------------


def test_stage_hub_file_size_bytes_refuses_on_low_headroom(tmp_path, monkeypatch):
    """size_bytes >> free: the REAL assert bodies refuse BEFORE any download
    (a network call here is a test failure)."""

    def fail_download(**kwargs):
        raise AssertionError("network call past a failing headroom assert")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail_download)
    target = tmp_path / "dest" / "huge.bin"
    with pytest.raises(RuntimeError, match=r"disk-headroom.*hub-staging"):
        hub.stage_hub_file("org/data", "huge.bin", target, size_bytes=10**15)
    assert not target.exists()


def test_stage_hub_file_default_none_never_asserts(tmp_path, monkeypatch):
    """Default size_bytes=None keeps the legacy path byte-identical: the
    headroom helper is never invoked (spy) and the file stages."""
    asserts: list = []
    monkeypatch.setattr(hub, "_assert_stage_headroom", lambda *a, **k: asserts.append((a, k)))

    def fake_hf_hub_download(
        *, repo_id, filename, repo_type=None, revision=None, local_dir=None, token=None
    ):
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"payload")
        return str(p)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    target = tmp_path / "file.json"
    assert hub.stage_hub_file("org/data", "file.json", target) == target
    assert asserts == []


def test_stage_hub_file_existing_target_skips_assert_even_with_size(tmp_path, monkeypatch):
    """An existing target (overwrite=False) returns FIRST — no assert, no
    network call, even when size_bytes is passed (skip-existing never asserts)."""
    asserts: list = []
    monkeypatch.setattr(hub, "_assert_stage_headroom", lambda *a, **k: asserts.append((a, k)))
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **k: (_ for _ in ()).throw(AssertionError("network call on existing target")),
    )
    target = tmp_path / "file.json"
    target.write_bytes(b"already staged")
    assert hub.stage_hub_file("org/data", "file.json", target, size_bytes=10**15) == target
    assert asserts == []


# ---------------------------------------------------------------------------
# Delegation: list_hf_files_under_path -> list_hf_entries_under_path
# ---------------------------------------------------------------------------


class _TreeApi:
    """Fake HfApi whose list_repo_tree yields REAL RepoFile entries (the
    test_hub.py `_repo_files` pattern) — the delegation must keep paths
    byte-identical to the historical implementation."""

    def __init__(self, entries=None, raises=None, file_exists_result=False):
        self._entries = entries or []
        self._raises = raises
        self._file_exists_result = file_exists_result
        self.file_exists_calls = 0

    def list_repo_tree(self, *, repo_id, repo_type, revision, recursive, path_in_repo):
        if self._raises is not None:
            raise self._raises
        return list(self._entries)

    def file_exists(self, repo_id, path, *, repo_type=None, revision=None):
        self.file_exists_calls += 1
        return self._file_exists_result


def test_list_files_delegates_and_returns_same_paths():
    api = _TreeApi(
        entries=[
            RepoFile(path="pfx/b.json", size=7, blob_id="b", oid="o"),
            RepoFile(path="pfx/a.json", size=3, blob_id="b", oid="o"),
        ]
    )
    assert hub.list_hf_files_under_path(api, "org/data", "pfx") == [
        "pfx/a.json",
        "pfx/b.json",
    ]
    assert hub.list_hf_entries_under_path(api, "org/data", "pfx") == [
        ("pfx/a.json", 3),
        ("pfx/b.json", 7),
    ]


def test_list_entries_exact_file_fallback_carries_none_size():
    """The tree endpoint 404s on file paths (#939): the file_exists fallback
    returns [(path, None)] — no size available on that branch (the accepted
    #2097 residual)."""
    from huggingface_hub.utils import EntryNotFoundError

    api = _TreeApi(raises=EntryNotFoundError("entry not found"), file_exists_result=True)
    assert hub.list_hf_entries_under_path(api, "org/data", "pfx/only.bin") == [
        ("pfx/only.bin", None)
    ]
    assert hub.list_hf_files_under_path(
        _TreeApi(raises=EntryNotFoundError("entry not found"), file_exists_result=True),
        "org/data",
        "pfx/only.bin",
    ) == ["pfx/only.bin"]


def test_list_entries_absent_path_returns_empty():
    from huggingface_hub.utils import EntryNotFoundError

    api = _TreeApi(raises=EntryNotFoundError("entry not found"), file_exists_result=False)
    assert hub.list_hf_entries_under_path(api, "org/data", "nope") == []
    assert hub.list_hf_files_under_path(api, "org/data", "nope") == []


def test_list_entries_empty_path_raises():
    with pytest.raises(ValueError, match="empty path"):
        hub.list_hf_entries_under_path(_TreeApi(), "org/data", "/")
    with pytest.raises(ValueError, match="empty path"):
        hub.list_hf_files_under_path(_TreeApi(), "org/data", "")
