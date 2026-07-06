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
    helpers resolve under one temp filesystem (no real repo writes).

    Also rebind ``task_workflow``'s OWN resolvers (#773): the active-CONSUMER
    reap gate walks the real ``tasks/`` tree via the ``list_by_status`` /
    ``tasks_dir`` ``ced`` imported from ``task_workflow``, which resolve the
    CACHED ``task_workflow.repo_root`` — NOT ``ced.repo_root``. Without rebinding
    them the gate would scan the live repo's tasks and (correctly) skip a reap of
    ``#658`` because real active tasks reference ``data/issue_658/``, breaking
    these sandboxed parity tests. Rebinding to the (empty) tmp ``tasks/`` makes
    the gate a clean no-op so the parity tests exercise ONLY the nested-store
    gate, as before."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import explore_persona_space.task_workflow as tw

    tw.invalidate_cache()
    monkeypatch.setattr(tw, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tw, "tasks_dir", lambda: tmp_path / "tasks")
    monkeypatch.setattr(ced, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(ced, "tasks_dir", lambda: tmp_path / "tasks")
    # Determinism pin (#924): force the on-main resolution path so a
    # hypothetical fresh-clone-on-a-branch test runner cannot flip the probe.
    monkeypatch.setattr(ced, "_off_main_checkout_root", lambda: None)
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

    # HF totals 300 too (store/-component-anchored paths), but a.pt/b.pt sizes
    # are swapped — per-file fails.
    hf_sizes = {"x/store/a.pt": 200, "x/store/b.pt": 100}
    assert ced.nested_store_is_mirrored(store_dir, hf_sizes) is False

    # Exact per-file sizes present at store/-anchored paths -> passes.
    hf_ok = {"x/store/a.pt": 100, "y/store/b.pt": 200}
    assert ced.nested_store_is_mirrored(store_dir, hf_ok) is True


def test_nested_store_is_mirrored_none_is_keep():
    """None (HF listing failed) is fail-toward-keep => not mirrored."""
    assert ced.nested_store_is_mirrored(Path("/nonexistent/store"), None) is False


# ─── #679 BLOCKER #2: basename-collision must NOT license a reap ──────────────


@pytest.mark.parametrize(
    "hf_path, expect_reaped",
    [
        # Path-faithful match: an HF path ending in store/runA/result.pt at the
        # same size IS a real mirror -> the wholesale reap proceeds.
        ("issue907_run/store/runA/result.pt", True),
        # COLLISION: an UNRELATED HF file shares ONLY the basename + size
        # (other/result.pt, no path overlap). The OLD basename-keyed check
        # falsely classified this as mirrored, after which rmtree(hf_dl) deleted
        # the non-re-downloadable store. The path-faithful check must SKIP it.
        ("other/result.pt", False),
    ],
)
def test_basename_collision_does_not_reap_unrelated_store(
    fake_repo, monkeypatch, hf_path, expect_reaped
):
    """A generated nested-store file ``runA/result.pt`` must be classified
    mirrored ONLY by a path-faithful match, never by a same-basename+size HF
    file at an UNRELATED path (#679 BLOCKER #2). On the collision case the reap
    is SKIPPED and the generated data survives; on the real path-faithful match
    the reap proceeds."""
    data_root = fake_repo / "data"
    issue_dir = data_root / "issue_907"
    nested = issue_dir / "hf_dl" / "store" / "runA"
    nested.mkdir(parents=True)
    (nested / "result.pt").write_bytes(b"z" * 1234)
    # Also a normal downloadable blob alongside the mis-rooted store.
    (issue_dir / "hf_dl" / "downloads").mkdir(parents=True)
    (issue_dir / "hf_dl" / "downloads" / "blob.bin").write_bytes(b"x" * 64)

    # HF lists EXACTLY ONE file at the basename "result.pt" of size 1234 — but
    # at the path under test (either path-faithful or an unrelated collision).
    monkeypatch.setattr(
        ced,
        "_hf_file_sizes",
        lambda repo, revision="main": {hf_path: 1234},
    )

    res = ced.clean_issue_downloads(907, apply=True, data_root=data_root)

    if expect_reaped:
        assert res.removed == ["data/issue_907/hf_dl"]
        assert res.skipped == []
        assert not (issue_dir / "hf_dl").exists()
        assert _read_sidecar(fake_repo) == []
    else:
        # Collision -> NOT verifiably mirrored -> SKIPPED, generated data kept.
        assert res.removed == []
        assert [name for name, _ in res.skipped] == ["data/issue_907/hf_dl"]
        assert (issue_dir / "hf_dl" / "store" / "runA" / "result.pt").exists()
        rows = _read_sidecar(fake_repo)
        assert len(rows) == 1
        assert rows[0]["kind"] == "nested-store-reap-skipped"
        assert rows[0]["task"] == 907


def test_basename_collision_unit_predicate(fake_repo):
    """Unit-level: only a PATH-FAITHFUL ``store/<rel>`` suffix match (at the
    matching size) licenses a reap. A bare-basename collision — even a single
    unrelated same-name+size entry — never does (#679 BLOCKER #2: no basename
    fallback). The store-internal subpath is preserved verbatim under
    ``issue<N>_<slug>/store/`` on HF, so the suffix match catches every real
    mirror without the collision risk a basename match carries."""
    data_root = fake_repo / "data"
    store_dir = data_root / "issue_908" / "hf_dl" / "store" / "runA"
    store_dir.mkdir(parents=True)
    (store_dir / "result.pt").write_bytes(b"z" * 1234)
    store_root = data_root / "issue_908" / "hf_dl" / "store"

    # Path-faithful suffix at matching size -> mirrored.
    assert ced.nested_store_is_mirrored(store_root, {"issue908/store/runA/result.pt": 1234}) is True
    # An UNRELATED single same-name+size entry at a NON-matching path is NOT a
    # mirror -> fail-toward-keep (no basename fallback).
    assert ced.nested_store_is_mirrored(store_root, {"other/result.pt": 1234}) is False
    # Right basename + RIGHT path but WRONG size -> not a match either.
    assert (
        ced.nested_store_is_mirrored(store_root, {"issue908/store/runA/result.pt": 9999}) is False
    )


# ─── #679 component-boundary BLOCKER: the match must anchor at a real ──────────
# ───   `store/` path component, never a bare `/<rel>` suffix nor an   ──────────
# ───   unbounded `store/<rel>` endswith that a `notstore` prefix can sneak past.


def test_single_segment_bare_suffix_collision_skips(fake_repo, monkeypatch):
    """Local ``store/result.pt`` (single-segment rel) vs an unrelated HF
    ``other/result.pt`` at the SAME size. The retired bare-``/<rel>`` suffix
    branch made ``other/result.pt`` end in ``/result.pt`` => a false mirror,
    licensing ``rmtree(hf_dl)`` on generated data. The component-anchored check
    must classify it NOT mirrored (unit) and SKIP the reap (integration)."""
    # Unit-level predicate.
    assert ced._local_file_is_mirrored("result.pt", 1234, {"other/result.pt": 1234}) is False
    # A real store-root mirror at the same single-segment rel DOES match.
    assert ced._local_file_is_mirrored("result.pt", 1234, {"store/result.pt": 1234}) is True

    # Integration-level reap: the mis-rooted single-segment store is SKIPPED.
    data_root = fake_repo / "data"
    issue_dir = data_root / "issue_910"
    nested = issue_dir / "hf_dl" / "store"
    nested.mkdir(parents=True)
    (nested / "result.pt").write_bytes(b"z" * 1234)
    monkeypatch.setattr(
        ced, "_hf_file_sizes", lambda repo, revision="main": {"other/result.pt": 1234}
    )
    res = ced.clean_issue_downloads(910, apply=True, data_root=data_root)
    assert res.removed == []
    assert [name for name, _ in res.skipped] == ["data/issue_910/hf_dl"]
    assert (issue_dir / "hf_dl" / "store" / "result.pt").exists()


def test_tail_collision_bare_suffix_skips(fake_repo, monkeypatch):
    """Local ``store/runA/result.pt`` vs an unrelated HF
    ``unrelated/runA/result.pt`` at the SAME size. The retired bare-``/<rel>``
    suffix branch tail-matched ``.../runA/result.pt`` => a false mirror. The
    component-anchored check must SKIP (no ``store/`` component on the HF
    side)."""
    assert (
        ced._local_file_is_mirrored("runA/result.pt", 1234, {"unrelated/runA/result.pt": 1234})
        is False
    )

    data_root = fake_repo / "data"
    issue_dir = data_root / "issue_911"
    nested = issue_dir / "hf_dl" / "store" / "runA"
    nested.mkdir(parents=True)
    (nested / "result.pt").write_bytes(b"z" * 1234)
    monkeypatch.setattr(
        ced, "_hf_file_sizes", lambda repo, revision="main": {"unrelated/runA/result.pt": 1234}
    )
    res = ced.clean_issue_downloads(911, apply=True, data_root=data_root)
    assert res.removed == []
    assert [name for name, _ in res.skipped] == ["data/issue_911/hf_dl"]
    assert (issue_dir / "hf_dl" / "store" / "runA" / "result.pt").exists()


def test_notstore_component_boundary_collision_skips(fake_repo, monkeypatch):
    """Local ``store/runA/result.pt`` vs HF ``issue/notstore/runA/result.pt`` at
    the SAME size. The retired unbounded ``store/<rel>`` ``endswith`` matched
    because ``notstore`` ends in ``store`` — a component-boundary miss. The
    anchored ``/store/<rel>`` check must SKIP (``notstore`` is not the ``store``
    component)."""
    assert (
        ced._local_file_is_mirrored("runA/result.pt", 1234, {"issue/notstore/runA/result.pt": 1234})
        is False
    )

    data_root = fake_repo / "data"
    issue_dir = data_root / "issue_912"
    nested = issue_dir / "hf_dl" / "store" / "runA"
    nested.mkdir(parents=True)
    (nested / "result.pt").write_bytes(b"z" * 1234)
    monkeypatch.setattr(
        ced,
        "_hf_file_sizes",
        lambda repo, revision="main": {"issue/notstore/runA/result.pt": 1234},
    )
    res = ced.clean_issue_downloads(912, apply=True, data_root=data_root)
    assert res.removed == []
    assert [name for name, _ in res.skipped] == ["data/issue_912/hf_dl"]
    assert (issue_dir / "hf_dl" / "store" / "runA" / "result.pt").exists()


def test_legitimate_store_at_repo_root_reaps(fake_repo, monkeypatch):
    """A real mirror with ``store/`` at the HF repo ROOT
    (``store/runA/result.pt``) is component-anchored (== ``store/<rel>``) =>
    mirrored (unit) and the wholesale reap proceeds (integration)."""
    assert (
        ced._local_file_is_mirrored("runA/result.pt", 1234, {"store/runA/result.pt": 1234}) is True
    )

    data_root = fake_repo / "data"
    issue_dir = data_root / "issue_913"
    nested = issue_dir / "hf_dl" / "store" / "runA"
    nested.mkdir(parents=True)
    (nested / "result.pt").write_bytes(b"z" * 1234)
    monkeypatch.setattr(
        ced, "_hf_file_sizes", lambda repo, revision="main": {"store/runA/result.pt": 1234}
    )
    res = ced.clean_issue_downloads(913, apply=True, data_root=data_root)
    assert res.removed == ["data/issue_913/hf_dl"]
    assert res.skipped == []
    assert not (issue_dir / "hf_dl").exists()
    assert _read_sidecar(fake_repo) == []


def test_legitimate_store_under_parent_dir_reaps(fake_repo, monkeypatch):
    """The DOMINANT production layout: ``issue<N>_<slug>/store/<rel>``. The
    ``store`` segment is anchored after a ``/`` (``.../store/runA/result.pt``)
    => mirrored (unit) and the wholesale reap proceeds (integration)."""
    assert (
        ced._local_file_is_mirrored(
            "runA/result.pt", 1234, {"issue679_X/store/runA/result.pt": 1234}
        )
        is True
    )

    data_root = fake_repo / "data"
    issue_dir = data_root / "issue_914"
    nested = issue_dir / "hf_dl" / "store" / "runA"
    nested.mkdir(parents=True)
    (nested / "result.pt").write_bytes(b"z" * 1234)
    monkeypatch.setattr(
        ced,
        "_hf_file_sizes",
        lambda repo, revision="main": {"issue914_X/store/runA/result.pt": 1234},
    )
    res = ced.clean_issue_downloads(914, apply=True, data_root=data_root)
    assert res.removed == ["data/issue_914/hf_dl"]
    assert res.skipped == []
    assert not (issue_dir / "hf_dl").exists()
    assert _read_sidecar(fake_repo) == []


# ─── #681 path transparency: cleaner resolves worktree data on the data disk ──


def test_cleaner_resolves_worktree_data_on_data_disk(fake_repo, monkeypatch):
    """After #681 the worktree data lives on the bind-mounted data disk, but the
    cleaner resolves the SAME relative path transparently — pointing the
    ``data_root=`` seam at a tmp_path that simulates the bind target reaps the
    ``hf_dl``/``g*_dl`` caches and KEEPS ``store/`` exactly as before. This is
    the path-transparency proof: the cleaner is indifferent to which physical
    device backs the resolved path (the bind/symlink contract, plan §1 #4)."""
    # `data_root` simulates the bind target: physically a different dir (as the
    # data disk is a different device), same logical role as the worktree data.
    data_disk_root = fake_repo / "mnt" / "eps-data" / "worktrees" / "issue-658" / "data"
    issue_dir = _plain_cache(data_disk_root, 658)
    res = ced.clean_issue_downloads(658, apply=True, data_root=data_disk_root)
    # Both re-downloadable caches reaped, regardless of the backing device.
    assert sorted(res.removed) == [
        "mnt/eps-data/worktrees/issue-658/data/issue_658/g1_dl",
        "mnt/eps-data/worktrees/issue-658/data/issue_658/hf_dl",
    ]
    assert res.failed == []
    assert not (issue_dir / "hf_dl").exists()
    assert not (issue_dir / "g1_dl").exists()
    # store/ — the durable, NOT re-downloadable artifact — is KEPT.
    assert (issue_dir / "store" / "generated.pt").exists()
