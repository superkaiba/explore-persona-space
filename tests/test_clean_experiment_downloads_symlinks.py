"""Tests for the symlinked-cache disposition logic in
``scripts/clean_experiment_downloads.py`` + the ``vm_disk_guard.py`` tier-(b)
reporting (task #915).

``shutil.rmtree`` refuses a symlink by design, so the #681-era relocation
pattern (``data/issue_<N>/hf_dl -> /mnt/eps-data/.../issue_<N>/hf_dl``) made
every janitor reap FAIL on the healthiest configuration. The fix reaps the
RESOLVED target only when it is verifiably the issue's relocated cache —
strictly inside the managed data-disk root (``EPS_VM_DATA_DISK_PATH``), a path
component naming the OWNING issue, and a directory whose basename equals the
cache name — and keeps (fail-toward-keep) anything else. A symlinked PARENT
``issue_<N>`` dir routes through the same disposition and its shared link is
NEVER unlinked. Dangling links are discovered + unlinked.

Both scripts live under ``scripts/`` (not an importable package), so they are
loaded via importlib exactly like ``tests/test_vm_disk_guard.py`` (ced first —
vm_disk_guard imports it by module name at load time). All fixtures live under
``tmp_path``; the managed data disk is faked via ``EPS_VM_DATA_DISK_PATH``.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPTS / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # register before exec (dataclass + future annotations)
    spec.loader.exec_module(mod)
    return mod


# clean_experiment_downloads must be importable by name before vm_disk_guard
# executes its top-level `from clean_experiment_downloads import ...`.
ced = _load("clean_experiment_downloads")
vdg = _load("vm_disk_guard")


# ─── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """Point ``ced.repo_root()`` at a temp dir (sidecar + rel-name helpers) and
    rebind ``task_workflow``'s resolvers so the active-CONSUMER gate walks the
    (empty) tmp ``tasks/`` tree instead of the live repo — a real active task
    referencing ``data/issue_761/`` would otherwise (correctly) SKIP the reap
    and break these sandboxed tests (same pattern as
    ``tests/test_clean_experiment_downloads_parity.py``)."""
    # syspath_prepend (not a bare sys.path.insert) so the entry is restored
    # at teardown instead of accumulating once per test.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
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


@pytest.fixture
def disk_root(fake_repo, monkeypatch) -> Path:
    """A fake managed data disk under tmp, wired in via the REAL env override
    (``data_disk_root()`` reads ``EPS_VM_DATA_DISK_PATH``; matches the
    env-override pattern in tests/test_vm_disk_guard_data_disk.py)."""
    root = fake_repo / "eps-data-disk"
    root.mkdir()
    monkeypatch.setenv("EPS_VM_DATA_DISK_PATH", str(root))
    return root


def _make_symlinked_issue(data_root: Path, disk_root: Path, issue_n: int = 761) -> Path:
    """data/issue_<N>/{hf_dl,g1_dl} as SYMLINKS to real caches under a fake
    data disk at disk_root/user/eps-data/issue_<N>/ (the live #761 shape)."""
    tgt_base = disk_root / "user" / "eps-data" / f"issue_{issue_n}"
    issue_dir = data_root / f"issue_{issue_n}"
    issue_dir.mkdir(parents=True)
    for name in ("hf_dl", "g1_dl"):
        tgt = tgt_base / name
        tgt.mkdir(parents=True)
        (tgt / "blob.bin").write_bytes(b"x" * 1024)
        (issue_dir / name).symlink_to(tgt)
    return issue_dir


def _read_sidecar(repo: Path) -> list[dict]:
    path = repo / ".claude" / "cache" / "disk-guard-events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# ─── 1: managed symlinked caches reaped (target + link both gone) ─────────────


def test_symlinked_cache_inside_managed_root_reaped(fake_repo, disk_root):
    data_root = fake_repo / "data"
    issue_dir = _make_symlinked_issue(data_root, disk_root)
    # A store/ SIBLING on the fake data disk must survive the reap.
    disk_store = disk_root / "user" / "eps-data" / "issue_761" / "store"
    disk_store.mkdir(parents=True)
    (disk_store / "generated.json").write_text('{"kept": true}')

    res = ced.clean_issue_downloads(761, apply=True, data_root=data_root)

    assert sorted(res.removed) == ["data/issue_761/g1_dl", "data/issue_761/hf_dl"]
    assert res.failed == []
    assert res.bytes_freed > 0
    for name in ("hf_dl", "g1_dl"):
        assert not os.path.lexists(issue_dir / name)  # link gone
        assert not (disk_root / "user" / "eps-data" / "issue_761" / name).exists()
    assert (disk_store / "generated.json").exists()  # data-disk store untouched


# ─── 2: outside-root target -> link unlinked, target KEPT ─────────────────────


def test_symlinked_cache_outside_managed_root_unlinked_target_kept(fake_repo, disk_root):
    data_root = fake_repo / "data"
    issue_dir = data_root / "issue_761"
    issue_dir.mkdir(parents=True)
    # The external target path ALSO contains an issue_761 component AND is a
    # cache-shaped dir (hf_dl), so ONLY the containment check keeps it — a
    # dropped containment check fails this test, not just the issue-naming
    # or cache-shape checks.
    ext = fake_repo / "external" / "issue_761" / "hf_dl"
    ext.mkdir(parents=True)
    (ext / "blob.bin").write_bytes(b"x" * 2048)
    (issue_dir / "hf_dl").symlink_to(ext)

    res = ced.clean_issue_downloads(761, apply=True, data_root=data_root)

    assert not os.path.lexists(issue_dir / "hf_dl")  # link gone
    assert (ext / "blob.bin").exists()  # target intact
    assert res.symlink_external_kept == [("data/issue_761/hf_dl", str(ext))]
    assert res.removed == []
    assert res.failed == []
    assert res.bytes_freed == 0  # external-kept never counts as freed

    rows = _read_sidecar(fake_repo)
    assert len(rows) == 1
    assert rows[0]["kind"] == "symlink-external-target-kept"
    assert rows[0]["task"] == 761
    assert rows[0]["path"] == "data/issue_761/hf_dl"
    assert rows[0]["target"] == str(ext)
    assert rows[0]["via"] == "link"


# ─── 3: inside root but naming ANOTHER issue -> treated external ──────────────


def test_symlink_inside_root_naming_other_issue_kept(fake_repo, disk_root):
    data_root = fake_repo / "data"
    issue_dir = data_root / "issue_761"
    issue_dir.mkdir(parents=True)
    other = disk_root / "user" / "eps-data" / "issue_999" / "hf_dl"
    other.mkdir(parents=True)
    (other / "blob.bin").write_bytes(b"x" * 1024)
    (issue_dir / "hf_dl").symlink_to(other)

    res = ced.clean_issue_downloads(761, apply=True, data_root=data_root)

    assert not os.path.lexists(issue_dir / "hf_dl")  # link gone
    assert (other / "blob.bin").exists()  # cross-issue target intact
    assert res.symlink_external_kept == [("data/issue_761/hf_dl", str(other))]
    assert res.removed == []
    assert res.failed == []


# ─── 4: dangling symlink -> discovered + unlinked ─────────────────────────────


def test_dangling_symlink_unlinked(fake_repo, disk_root):
    data_root = fake_repo / "data"
    issue_dir = data_root / "issue_761"
    issue_dir.mkdir(parents=True)
    gone = disk_root / "user" / "eps-data" / "issue_761" / "hf_dl"  # never created
    (issue_dir / "hf_dl").symlink_to(gone)

    # Discovery-filter regression: is_dir() alone would drop the dangling link.
    caches = ced.download_cache_dirs(761, data_root=data_root)
    assert caches == [issue_dir / "hf_dl"]

    res = ced.clean_issue_downloads(761, apply=True, data_root=data_root)

    assert res.removed == ["data/issue_761/hf_dl"]
    assert res.failed == []
    assert res.sizes_bytes["data/issue_761/hf_dl"] == 0
    assert res.symlink_targets["data/issue_761/hf_dl"] == ""
    assert not os.path.lexists(issue_dir / "hf_dl")


# ─── 5: dry-run is disposition-aware and removes nothing ──────────────────────


def test_symlink_dry_run_reports_and_removes_nothing(fake_repo, disk_root):
    data_root = fake_repo / "data"
    issue_dir = data_root / "issue_761"
    issue_dir.mkdir(parents=True)
    managed = disk_root / "user" / "eps-data" / "issue_761" / "hf_dl"
    managed.mkdir(parents=True)
    (managed / "blob.bin").write_bytes(b"x" * 1024)
    (issue_dir / "hf_dl").symlink_to(managed)
    ext = fake_repo / "external" / "issue_761" / "g1_dl"
    ext.mkdir(parents=True)
    (ext / "blob.bin").write_bytes(b"x" * 1024)
    (issue_dir / "g1_dl").symlink_to(ext)

    res = ced.clean_issue_downloads(761, apply=False, data_root=data_root)

    # Links + targets ALL intact after a dry run.
    for link, tgt in ((issue_dir / "hf_dl", managed), (issue_dir / "g1_dl", ext)):
        assert os.path.lexists(link) and link.is_symlink()
        assert (tgt / "blob.bin").exists()
    assert res.removed == ["data/issue_761/hf_dl"]  # would-remove (managed)
    assert res.symlink_external_kept == [("data/issue_761/g1_dl", str(ext))]
    assert res.failed == []
    assert res.symlink_targets == {
        "data/issue_761/hf_dl": str(managed),
        "data/issue_761/g1_dl": str(ext),
    }


# ─── 6: the nested-store parity gate still protects THROUGH the link ──────────


def test_parity_gate_blocks_reap_through_symlink(fake_repo, disk_root, monkeypatch):
    data_root = fake_repo / "data"
    issue_dir = _make_symlinked_issue(data_root, disk_root)
    # A store/ NESTED inside the relocated hf_dl cache (mis-rooted run).
    nested = disk_root / "user" / "eps-data" / "issue_761" / "hf_dl" / "store"
    nested.mkdir(parents=True)
    (nested / "generated.json").write_text('{"kept": true}')
    # Nothing mirrored on HF -> the parity gate must SKIP the hf_dl reap.
    monkeypatch.setattr(ced, "_hf_file_sizes", lambda repo, revision="main": {})

    res = ced.clean_issue_downloads(761, apply=True, data_root=data_root)

    assert [name for name, _ in res.skipped] == ["data/issue_761/hf_dl"]
    assert os.path.lexists(issue_dir / "hf_dl") and (issue_dir / "hf_dl").is_symlink()
    assert (nested / "generated.json").exists()  # resolved contents protected
    # A gate-2-SKIPped link never reaches the symlink branch.
    assert "data/issue_761/hf_dl" not in res.symlink_targets
    # The store-less g1_dl sibling still reaps normally.
    assert res.removed == ["data/issue_761/g1_dl"]
    assert res.failed == []


# ─── 7: the exact incident regression, at the tier-(b) call site ──────────────


def test_tier_b_symlinked_cache_no_failed_row(fake_repo, disk_root, monkeypatch):
    data_root = fake_repo / "data"
    issue_dir = _make_symlinked_issue(data_root, disk_root)
    monkeypatch.setattr(vdg, "_resolve_issue_status", lambda n: "completed")

    res = vdg.clean_terminal_download_caches(apply=True, data_root=data_root)

    assert not any("FAILED to remove" in d for d in res.detail)
    assert res.bytes_freed > 0
    for name in ("hf_dl", "g1_dl"):
        assert not os.path.lexists(issue_dir / name)
        assert not (disk_root / "user" / "eps-data" / "issue_761" / name).exists()


# ─── 8: idempotent — a second apply run finds nothing ─────────────────────────


def test_symlink_reap_idempotent(fake_repo, disk_root):
    data_root = fake_repo / "data"
    _make_symlinked_issue(data_root, disk_root)

    first = ced.clean_issue_downloads(761, apply=True, data_root=data_root)
    assert len(first.removed) == 2

    second = ced.clean_issue_downloads(761, apply=True, data_root=data_root)
    assert second.removed == []
    assert second.failed == []
    assert second.symlink_external_kept == []


# ─── 9 (MF1): same-issue alias to a store/ dir -> KEPT ────────────────────────


def test_symlink_to_store_target_kept(fake_repo, disk_root):
    data_root = fake_repo / "data"
    issue_dir = data_root / "issue_761"
    issue_dir.mkdir(parents=True)
    # Inside the managed root AND naming the owning issue — but the target is
    # a store/ (basename != cache name): only the MF1 cache-shape predicate
    # keeps it (gate 2's rglob matches descendants, never the resolved root).
    store = disk_root / "user" / "eps-data" / "issue_761" / "store"
    store.mkdir(parents=True)
    (store / "generated.json").write_text('{"kept": true}')
    (issue_dir / "hf_dl").symlink_to(store)

    res = ced.clean_issue_downloads(761, apply=True, data_root=data_root)

    assert (store / "generated.json").exists()  # store tree INTACT
    assert res.symlink_external_kept == [("data/issue_761/hf_dl", str(store))]
    assert res.removed == []
    assert res.failed == []
    assert not os.path.lexists(issue_dir / "hf_dl")  # link unlinked


# ─── 10 (MF1): symlink to a FILE (even cache-named) -> KEPT ───────────────────


def test_symlink_to_file_target_kept(fake_repo, disk_root):
    data_root = fake_repo / "data"
    issue_dir = data_root / "issue_761"
    issue_dir.mkdir(parents=True)
    # Managed root + owning issue + exact cache NAME — but a FILE, not a dir.
    tgt_dir = disk_root / "user" / "eps-data" / "issue_761"
    tgt_dir.mkdir(parents=True)
    tgt_file = tgt_dir / "g1_dl"
    tgt_file.write_bytes(b"x" * 512)
    (issue_dir / "g1_dl").symlink_to(tgt_file)

    res = ced.clean_issue_downloads(761, apply=True, data_root=data_root)

    assert tgt_file.exists()  # file INTACT
    assert res.symlink_external_kept == [("data/issue_761/g1_dl", str(tgt_file))]
    assert res.removed == []
    assert res.failed == []
    assert not os.path.lexists(issue_dir / "g1_dl")  # link unlinked


# ─── 11 (MF2): symlinked PARENT, external target -> nothing touched ───────────


def test_symlinked_parent_external_target_kept(fake_repo, disk_root):
    data_root = fake_repo / "data"
    data_root.mkdir(parents=True)
    ext_issue = fake_repo / "external" / "issue_761"
    hf_dl = ext_issue / "hf_dl"
    hf_dl.mkdir(parents=True)
    (hf_dl / "blob.bin").write_bytes(b"x" * 1024)
    parent_link = data_root / "issue_761"
    parent_link.symlink_to(ext_issue)

    res = ced.clean_issue_downloads(761, apply=True, data_root=data_root)

    assert (hf_dl / "blob.bin").exists()  # external content INTACT
    assert parent_link.is_symlink()  # the shared parent link is NEVER unlinked
    assert res.symlink_external_kept == [("data/issue_761/hf_dl", str(hf_dl))]
    assert res.removed == []
    assert res.failed == []

    rows = _read_sidecar(fake_repo)
    assert len(rows) == 1
    assert rows[0]["kind"] == "symlink-external-target-kept"
    assert rows[0]["via"] == "parent"


# ─── 12 (MF2): symlinked PARENT, managed target -> caches reaped, link kept ───


def test_symlinked_parent_managed_reaped(fake_repo, disk_root):
    data_root = fake_repo / "data"
    data_root.mkdir(parents=True)
    disk_issue = disk_root / "user" / "eps-data" / "issue_761"
    for name in ("hf_dl", "g1_dl"):
        d = disk_issue / name
        d.mkdir(parents=True)
        (d / "blob.bin").write_bytes(b"x" * 1024)
    parent_link = data_root / "issue_761"
    parent_link.symlink_to(disk_issue)

    res = ced.clean_issue_downloads(761, apply=True, data_root=data_root)

    assert sorted(res.removed) == ["data/issue_761/g1_dl", "data/issue_761/hf_dl"]
    assert res.failed == []
    assert res.bytes_freed > 0
    for name in ("hf_dl", "g1_dl"):
        assert not (disk_issue / name).exists()  # resolved caches GONE
    assert parent_link.is_symlink()  # parent link INTACT


# ─── 13: target rmtree failure -> link survives (deletion-order crash safety) ─


def test_target_rmtree_failure_link_survives(fake_repo, disk_root, monkeypatch):
    data_root = fake_repo / "data"
    issue_dir = _make_symlinked_issue(data_root, disk_root)

    def _boom(path, *a, **k):
        raise OSError(f"simulated rmtree failure on {path}")

    monkeypatch.setattr(ced.shutil, "rmtree", _boom)

    res = ced.clean_issue_downloads(761, apply=True, data_root=data_root)

    assert sorted(res.failed) == ["data/issue_761/g1_dl", "data/issue_761/hf_dl"]
    assert res.removed == []
    for name in ("hf_dl", "g1_dl"):
        # The link survives a target-reap failure (target-before-link order),
        # so the next run re-discovers and retries.
        assert os.path.lexists(issue_dir / name)
        assert (disk_root / "user" / "eps-data" / "issue_761" / name / "blob.bin").exists()
    # res.failed non-empty is the CLI exit-2 driver.
    assert res.failed


# ─── 14 (MF2 round-2 regression): DOUBLE link — parent AND child both links ───
# Round-1 computed `parent_linked = (not cache_dir.is_symlink()) and
# cache_dir.parent.is_symlink()`, so when the parent issue dir was a link AND
# the cache entry inside its target tree was ITSELF a link,
# cache_dir.is_symlink() was True through the parent link, parent_linked came
# out False, the case classified as a DIRECT link, and apply-mode unlink()
# resolved through the shared parent link and removed the child entry INSIDE
# the external unmanaged tree — violating the plan §2 MF2 row "symlinked
# parent + external target: SKIP entirely — nothing deleted, nothing
# unlinked". Concern id: double-link-unlink-through-parent-mf2.


def test_double_link_parent_and_child_external_target_kept(fake_repo, disk_root):
    data_root = fake_repo / "data"
    data_root.mkdir(parents=True)
    # data/issue_761 -> external tree, whose hf_dl entry is ITSELF a symlink
    # to another external dir (the double-link topology).
    ext_issue = fake_repo / "external" / "issue_761"
    ext_issue.mkdir(parents=True)
    ext_target = fake_repo / "external" / "real_hf_dl"
    ext_target.mkdir(parents=True)
    (ext_target / "blob.bin").write_bytes(b"x" * 1024)
    (ext_issue / "hf_dl").symlink_to(ext_target)
    parent_link = data_root / "issue_761"
    parent_link.symlink_to(ext_issue)

    res = ced.clean_issue_downloads(761, apply=True, data_root=data_root)

    # BOTH links survive: the shared parent link AND the child link entry
    # inside the external tree (round-1 unlinked the child THROUGH the
    # parent link). The external tree is byte-untouched.
    assert parent_link.is_symlink()
    assert os.path.lexists(ext_issue / "hf_dl")
    assert (ext_issue / "hf_dl").is_symlink()
    assert (ext_target / "blob.bin").exists()
    assert res.symlink_external_kept == [("data/issue_761/hf_dl", str(ext_target))]
    assert res.removed == []
    assert res.failed == []

    rows = _read_sidecar(fake_repo)
    assert len(rows) == 1
    assert rows[0]["kind"] == "symlink-external-target-kept"
    # Parent-link ownership dominates: via is "parent" even though the child
    # entry is itself a link.
    assert rows[0]["via"] == "parent"


def test_double_link_parent_and_child_dangling_kept(fake_repo, disk_root):
    """Double-link variant with a DANGLING child target: nothing to reap and
    the child entry inside the parent's target tree is not ours to unlink —
    kept + sidecar-escalated, never unlinked."""
    data_root = fake_repo / "data"
    data_root.mkdir(parents=True)
    ext_issue = fake_repo / "external" / "issue_761"
    ext_issue.mkdir(parents=True)
    (ext_issue / "hf_dl").symlink_to(fake_repo / "external" / "gone")  # dangling
    parent_link = data_root / "issue_761"
    parent_link.symlink_to(ext_issue)

    res = ced.clean_issue_downloads(761, apply=True, data_root=data_root)

    assert parent_link.is_symlink()
    assert os.path.lexists(ext_issue / "hf_dl")  # dangling child link NOT unlinked
    assert [rel for rel, _ in res.symlink_external_kept] == ["data/issue_761/hf_dl"]
    assert res.removed == []
    assert res.failed == []

    rows = _read_sidecar(fake_repo)
    assert len(rows) == 1
    assert rows[0]["via"] == "parent"


def test_double_link_managed_parent_child_link_target_reaped_links_survive(fake_repo, disk_root):
    """Managed variant of the double-link topology (parent link resolves
    INSIDE the managed data disk; the child entry there is itself a link to a
    fully-validated managed cache dir): the resolved TARGET is reaped, but
    neither link is unlinked — the parent link is shared and the child link
    entry lives inside its target tree."""
    data_root = fake_repo / "data"
    data_root.mkdir(parents=True)
    disk_issue = disk_root / "user" / "eps-data" / "issue_761"
    real = disk_issue / "real" / "hf_dl"  # names issue_761, cache-shaped dir
    real.mkdir(parents=True)
    (real / "blob.bin").write_bytes(b"x" * 1024)
    (disk_issue / "hf_dl").symlink_to(real)
    parent_link = data_root / "issue_761"
    parent_link.symlink_to(disk_issue)

    res = ced.clean_issue_downloads(761, apply=True, data_root=data_root)

    assert res.removed == ["data/issue_761/hf_dl"]
    assert res.failed == []
    assert not real.exists()  # fully-validated managed target reaped
    assert parent_link.is_symlink()  # parent link intact
    # The child link entry inside the managed tree is NOT unlinked through
    # the parent link (left dangling in the managed tree).
    assert os.path.lexists(disk_issue / "hf_dl")


# ─── 17: direct link, renamed-basename / eval_results targets kept ────────────


def test_symlink_renamed_basename_target_kept(fake_repo, disk_root):
    """A direct link whose managed-root target has a DIFFERENT basename
    (hf_dl -> .../issue_761/hf_dl_old) fails the name-preserving cache-shape
    prong: link unlinked, target KEPT."""
    data_root = fake_repo / "data"
    issue_dir = data_root / "issue_761"
    issue_dir.mkdir(parents=True)
    tgt = disk_root / "user" / "eps-data" / "issue_761" / "hf_dl_old"
    tgt.mkdir(parents=True)
    (tgt / "blob.bin").write_bytes(b"x" * 1024)
    (issue_dir / "hf_dl").symlink_to(tgt)

    res = ced.clean_issue_downloads(761, apply=True, data_root=data_root)

    assert (tgt / "blob.bin").exists()  # renamed target KEPT
    assert not os.path.lexists(issue_dir / "hf_dl")  # direct link unlinked
    assert res.symlink_external_kept == [("data/issue_761/hf_dl", str(tgt))]
    assert res.removed == []
    assert res.failed == []


def test_symlink_to_eval_results_target_kept(fake_repo, disk_root):
    """A direct link resolving to an eval_results/ dir (durable artifacts,
    never a cache) is kept via the basename prong (fail-toward-keep)."""
    data_root = fake_repo / "data"
    issue_dir = data_root / "issue_761"
    issue_dir.mkdir(parents=True)
    tgt = disk_root / "user" / "eps-data" / "issue_761" / "eval_results"
    tgt.mkdir(parents=True)
    (tgt / "metrics.json").write_text("{}")
    (issue_dir / "hf_dl").symlink_to(tgt)

    res = ced.clean_issue_downloads(761, apply=True, data_root=data_root)

    assert (tgt / "metrics.json").exists()  # durable artifacts KEPT
    assert not os.path.lexists(issue_dir / "hf_dl")
    assert res.symlink_external_kept == [("data/issue_761/hf_dl", str(tgt))]
    assert res.removed == []
    assert res.failed == []


# ─── 18: CLI exit-code contract (main() returns 2 iff res.failed) ─────────────


def test_cli_exit_code_2_on_failed_reap(fake_repo, disk_root, monkeypatch):
    data_root = fake_repo / "data"
    _make_symlinked_issue(data_root, disk_root)
    monkeypatch.setattr(ced, "_running_pod_side", lambda: False)

    def _boom(path, *a, **k):
        raise OSError(f"simulated rmtree failure on {path}")

    monkeypatch.setattr(ced.shutil, "rmtree", _boom)

    assert ced.main(["761", "--apply"]) == 2


def test_cli_exit_code_0_on_clean_run(fake_repo, disk_root, monkeypatch):
    monkeypatch.setattr(ced, "_running_pod_side", lambda: False)
    data_root = fake_repo / "data"
    _make_symlinked_issue(data_root, disk_root)

    assert ced.main(["761", "--apply"]) == 0
