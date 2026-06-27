"""Tests for the nested-``store/`` parity guard in
``scripts/clean_experiment_downloads.py`` (task #679).

A wholesale ``shutil.rmtree(hf_dl)`` at a terminal reap is safe for a normal
re-downloadable cache, but DESTROYS generated (NOT re-downloadable) data if a
``store/`` tree was mis-rooted UNDER the cache dir. The guard refuses the reap
unless every nested-store file is verifiably mirrored on HF (per-file size
match, fail-toward-keep), and escalates a SKIP to the shared disk-guard sidecar.

Covers three cases the plan names:
  * plain ``hf_dl`` cache (no nested store) — reaped exactly as before,
  * nested store whose files ARE mirrored on HF (matching size) — reaped,
  * nested store NOT verifiably mirrored (size mismatch / HF unavailable) —
    SKIPPED + escalated, the generated data preserved.

The script lives under ``scripts/`` (not an importable package), so it is
loaded via importlib exactly like ``tests/test_vm_disk_guard.py``. The HF Hub
API is faked with ``unittest.mock`` (no ``responses`` dep, no network).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPTS / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


ced = _load("clean_experiment_downloads")


# ─── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """Point ``ced.repo_root()`` at a temp dir so the sidecar + rel-name
    helpers resolve under one temp filesystem (no real repo writes)."""
    monkeypatch.setattr(ced, "repo_root", lambda: tmp_path)
    return tmp_path


def _plain_cache(data_root: Path, issue_n: int) -> Path:
    """A normal re-downloadable issue cache: hf_dl + g1_dl (each a file),
    plus a SIBLING store/ (kept by the keep/delete contract, untouched here)."""
    issue_dir = data_root / f"issue_{issue_n}"
    for cache in ("hf_dl", "g1_dl"):
        d = issue_dir / cache
        d.mkdir(parents=True)
        (d / "blob.bin").write_bytes(b"x" * 2048)
    sib_store = issue_dir / "store"
    sib_store.mkdir(parents=True)
    (sib_store / "generated.pt").write_bytes(b"y" * 4096)
    return issue_dir


def _nested_store_cache(data_root: Path, issue_n: int, *, store_size: int = 4096) -> Path:
    """An issue whose hf_dl cache has a ``store/`` MIS-ROOTED inside it (the
    anomalous case the guard defends). Returns the issue dir; the nested store
    file is ``v0_summaries.pt`` of ``store_size`` bytes."""
    issue_dir = data_root / f"issue_{issue_n}"
    hf_dl = issue_dir / "hf_dl"
    (hf_dl / "downloads").mkdir(parents=True)
    (hf_dl / "downloads" / "blob.bin").write_bytes(b"x" * 1024)
    nested = hf_dl / "store"
    nested.mkdir(parents=True)
    (nested / "v0_summaries.pt").write_bytes(b"z" * store_size)
    return issue_dir


def _read_sidecar(repo: Path) -> list[dict]:
    path = repo / ".claude" / "cache" / "disk-guard-events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# ─── case 1: plain re-downloadable cache reaped (no nested store) ─────────────


def test_plain_cache_reaped_no_hf_call(fake_repo, monkeypatch):
    """A cache with no nested store/ is reaped exactly as before, and the guard
    never reaches the HF listing (the expensive call is gated on a nested
    store/ being present)."""
    data_root = fake_repo / "data"
    issue_dir = _plain_cache(data_root, 901)

    # Any HF call here would be a bug — the guard must short-circuit.
    def _boom(*a, **k):
        raise AssertionError("HF listing must not be called when no nested store/ exists")

    monkeypatch.setattr(ced, "_hf_file_sizes", _boom)

    res = ced.clean_issue_downloads(901, apply=True, data_root=data_root)

    assert sorted(res.removed) == ["data/issue_901/g1_dl", "data/issue_901/hf_dl"]
    assert res.skipped == []
    assert res.failed == []
    assert not (issue_dir / "hf_dl").exists()
    assert not (issue_dir / "g1_dl").exists()
    # The SIBLING store/ is never touched (the keep/delete contract).
    assert (issue_dir / "store" / "generated.pt").exists()
    assert _read_sidecar(fake_repo) == []


# ─── case 2: nested store verifiably mirrored on HF -> reaped ─────────────────


def test_nested_store_mirrored_is_reaped(fake_repo, monkeypatch):
    """When every nested-store file is present on HF at a MATCHING size, the
    wholesale reap proceeds (the generated data survives on HF)."""
    data_root = fake_repo / "data"
    issue_dir = _nested_store_cache(data_root, 902, store_size=4096)

    # HF mirror: the same basename at the same size lives in the data repo.
    monkeypatch.setattr(
        ced,
        "_hf_file_sizes",
        lambda repo, revision="main": {"issue902_run/store/v0_summaries.pt": 4096},
    )

    res = ced.clean_issue_downloads(902, apply=True, data_root=data_root)

    assert res.removed == ["data/issue_902/hf_dl"]
    assert res.skipped == []
    assert not (issue_dir / "hf_dl").exists()
    assert _read_sidecar(fake_repo) == []


# ─── case 3: nested store NOT verifiably mirrored -> SKIPPED + escalated ──────


def test_nested_store_size_mismatch_is_skipped_and_escalated(fake_repo, monkeypatch):
    """A nested store whose file size differs from HF is NOT verifiably
    mirrored: the reap is SKIPPED (generated data kept) and an escalation row is
    written to the shared disk-guard sidecar."""
    data_root = fake_repo / "data"
    issue_dir = _nested_store_cache(data_root, 903, store_size=4096)

    # HF has the basename but at a DIFFERENT size — not a match.
    monkeypatch.setattr(
        ced,
        "_hf_file_sizes",
        lambda repo, revision="main": {"issue903_run/store/v0_summaries.pt": 9999},
    )

    res = ced.clean_issue_downloads(903, apply=True, data_root=data_root)

    assert res.removed == []
    assert [name for name, _ in res.skipped] == ["data/issue_903/hf_dl"]
    # The cache dir (and the generated store inside it) is preserved.
    assert (issue_dir / "hf_dl" / "store" / "v0_summaries.pt").exists()

    rows = _read_sidecar(fake_repo)
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "nested-store-reap-skipped"
    assert row["task"] == 903
    assert row["path"] == "data/issue_903/hf_dl"
    assert "ts" in row


def test_nested_store_hf_unavailable_is_skipped(fake_repo, monkeypatch):
    """HF listing failure (None) is fail-toward-keep: the reap is SKIPPED even
    though the data MIGHT be mirrored — never delete generated data we cannot
    positively confirm is preserved."""
    data_root = fake_repo / "data"
    issue_dir = _nested_store_cache(data_root, 904)
    monkeypatch.setattr(ced, "_hf_file_sizes", lambda repo, revision="main": None)

    res = ced.clean_issue_downloads(904, apply=True, data_root=data_root)

    assert res.removed == []
    assert [name for name, _ in res.skipped] == ["data/issue_904/hf_dl"]
    assert (issue_dir / "hf_dl" / "store" / "v0_summaries.pt").exists()
    assert len(_read_sidecar(fake_repo)) == 1


def test_dry_run_does_not_delete_and_does_not_write_sidecar(fake_repo, monkeypatch):
    """In dry-run, a skipped nested-store cache is reported but nothing is
    deleted and no sidecar row is persisted (apply=False reports only)."""
    data_root = fake_repo / "data"
    issue_dir = _nested_store_cache(data_root, 905, store_size=4096)
    monkeypatch.setattr(
        ced,
        "_hf_file_sizes",
        lambda repo, revision="main": {"x/store/v0_summaries.pt": 1},
    )

    res = ced.clean_issue_downloads(905, apply=False, data_root=data_root)

    assert res.removed == []
    assert [name for name, _ in res.skipped] == ["data/issue_905/hf_dl"]
    assert (issue_dir / "hf_dl").exists()  # nothing deleted in dry-run
    assert _read_sidecar(fake_repo) == []  # apply=False does not persist


# ─── unit coverage of the pure parity predicate ──────────────────────────────


def test_nested_store_is_mirrored_per_file_not_sum(fake_repo, monkeypatch):
    """The check is a PER-FILE size match, not a size-SUM: two files whose
    sizes sum to the same total but individually differ must NOT pass."""
    data_root = fake_repo / "data"
    issue_dir = data_root / "issue_906"
    nested = issue_dir / "hf_dl" / "store"
    nested.mkdir(parents=True)
    (nested / "a.pt").write_bytes(b"a" * 100)
    (nested / "b.pt").write_bytes(b"b" * 200)
    store_dir = issue_dir / "hf_dl" / "store"

    # HF totals 300 too, but a.pt/b.pt sizes are swapped — per-file fails.
    hf_sizes = {"x/a.pt": 200, "x/b.pt": 100}
    assert ced.nested_store_is_mirrored(store_dir, hf_sizes) is False

    # Exact per-file sizes present -> passes.
    hf_ok = {"x/a.pt": 100, "y/b.pt": 200}
    assert ced.nested_store_is_mirrored(store_dir, hf_ok) is True


def test_nested_store_is_mirrored_none_is_keep():
    """None (HF listing failed) is fail-toward-keep => not mirrored."""
    assert ced.nested_store_is_mirrored(Path("/nonexistent/store"), None) is False
