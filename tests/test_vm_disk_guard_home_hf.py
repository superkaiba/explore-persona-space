"""Tests for ``vm_disk_guard.py`` tier (e) — the home HF-hub stale-REVISION
trim (task #1377): ``clean_home_hf_stale_revisions`` keeps, per repo, the
newest revision + every ref'd revision and deletes only unref'd non-newest
revisions older than the age gate (default 7 d).

RECONCILIATION NOTE (#1376 + #1377): the tier these tests pin was landed
independently by both tasks and unified into ONE function keeping #1377's
names/knobs and the UNION of both KEEP protections. The arm-2 pins below are
unchanged; the fixtures gained the fields the unified tier also dereferences
(``repo_type`` / ``last_accessed`` / repo ``size_on_disk`` / revision
``files``) with values chosen so #1376's whole-stale-repo arm 1 and
exclusive-blob atime guard stay inert here (see ``_repo`` / ``_rev``). The
#1376 arms have their own suite: ``tests/test_vm_disk_guard_home_hf_cache.py``.

HERMETIC BY CONSTRUCTION: every test passes an explicit fixture
``cache_root`` (or monkeypatches the ``_scan_hf_cache`` seam to a fake
``HFCacheInfo``) — the REAL ``~/.cache/huggingface`` is never read or
written. The run_guard-level test stubs the tier out entirely except for a
``_boom`` recorder, mirroring ``tests/test_janitor_noncanonical_caches.py``.

Fixture note (discriminating power): the plan's single-repo 4-revision
fixture is internally inconsistent — a "newest-but-old" revision cannot have
the repo's max ``last_modified`` while a "young" sibling in the SAME repo is
younger than the cutoff. The fixture therefore splits into two repos so each
survival is attributable to exactly ONE keep rule:
  repo-a: newest-but-old (unref'd AND older than cutoff -> kept SOLELY by
          keep-newest), old+ref'd (non-newest AND older than cutoff -> kept
          SOLELY by keep-ref'd), old+unref'd (DELETED);
  repo-b: newest (kept: newest), young+unref'd non-newest (kept SOLELY by
          the age gate — test 3's ``max_age_days=0.0`` arm flips it stale);
  single: one old unref'd revision (kept: single revision == newest).

Loaded via importlib like ``tests/test_janitor_noncanonical_caches.py``
(ced first — vm_disk_guard imports it by module name at load time).
"""

import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPTS / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # register before exec (dataclass + future annotations)
    spec.loader.exec_module(mod)
    return mod


ced = _load("clean_experiment_downloads")
vdg = _load("vm_disk_guard")

DAY = 86400.0


# ─── fixtures / helpers ──────────────────────────────────────────────────────


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Point both modules' repo_root at a temp dir so the disk-guard sidecar
    resolves under it (same mechanism as test_janitor_noncanonical_caches)."""
    monkeypatch.setattr(ced, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(vdg, "repo_root", lambda: tmp_path)
    return tmp_path


def _read_sidecar(repo_path: Path) -> list[dict]:
    import json

    path = repo_path / ".claude" / "cache" / "disk-guard-events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _rev(commit_hash: str, last_modified: float, *, refs=(), size: int = 1000):
    """A signature-conformant fake CachedRevisionInfo (the fields tier (e)
    dereferences: commit_hash, refs frozenset, last_modified, size_on_disk,
    files — empty ``files`` means no exclusive blobs, so the #1376
    exclusive-blob atime keep-guard passes through, preserving the pure
    arm-2 semantics these tests pin)."""
    return SimpleNamespace(
        commit_hash=commit_hash,
        refs=frozenset(refs),
        last_modified=last_modified,
        size_on_disk=size,
        files=(),
    )


def _repo(repo_id: str, now: float, revisions: list):
    """A signature-conformant fake CachedRepoInfo. ``last_accessed=now`` so
    the #1376 whole-stale-repo arm 1 (repo-level ``last_accessed`` > window)
    NEVER fires in these fixtures — including test 3's ``max_age_days=0.0``
    arm (``now - now = 0`` is not ``> 0``) — keeping every assertion here a
    pure arm-2 (revision-level) pin, as #1377 designed them."""
    return SimpleNamespace(
        repo_id=repo_id,
        repo_type="model",
        last_accessed=now,
        size_on_disk=sum(r.size_on_disk for r in revisions),
        revisions=revisions,
    )


def _fake_home_cache_info(now: float, executed: list, requested: list, *, freed: int = 55555):
    """Fake HFCacheInfo with the discriminating fixture (module docstring)."""
    repo_a = _repo(
        "org/repo-a",
        now,
        [
            # newest in repo-a, UNREF'D and OLDER than the 7d cutoff — its
            # survival is attributable SOLELY to keep-newest.
            _rev("newest_old", now - 10 * DAY),
            # non-newest and OLDER than cutoff, but ref'd — survival
            # attributable SOLELY to keep-ref'd.
            _rev("old_refd", now - 20 * DAY, refs={"main"}),
            # non-newest, unref'd, older than cutoff — the ONE deletion target.
            _rev("old_unrefd", now - 30 * DAY, size=2000),
        ],
    )
    repo_b = _repo(
        "org/repo-b",
        now,
        [
            _rev("b_newest", now - 1 * DAY),  # newest — kept
            # non-newest, unref'd, YOUNGER than the 7d cutoff — survival
            # attributable SOLELY to the age gate (test 3 flips it stale
            # with max_age_days=0.0).
            _rev("b_young", now - 2 * DAY),
        ],
    )
    single = _repo(
        "org/single",
        now,
        # single revision == newest by construction: always kept, any age.
        [_rev("single_old", now - 40 * DAY)],
    )

    def delete_revisions(*hashes):
        requested.append(hashes)
        return SimpleNamespace(
            expected_freed_size=freed,
            execute=lambda: executed.append(hashes),
        )

    return SimpleNamespace(repos=[repo_a, repo_b, single], delete_revisions=delete_revisions)


@pytest.fixture
def home_cache(tmp_path, monkeypatch):
    """A fixture cache_root with a hub/ dir + the fake scan seam wired in.
    Returns (cache_root, now, executed, requested)."""
    (tmp_path / "hfcache" / "hub").mkdir(parents=True)
    now = time.time()
    executed: list = []
    requested: list = []
    monkeypatch.setattr(vdg, "_running_pod_side", lambda: False)
    monkeypatch.setattr(
        vdg, "_scan_hf_cache", lambda hub: _fake_home_cache_info(now, executed, requested)
    )
    return tmp_path / "hfcache", now, executed, requested


# ─── 1: keep-newest + keep-ref'd + age gate (durability pin) ─────────────────


def test_home_hf_tier_keeps_newest_and_refd(home_cache):
    cache_root, now, _executed, requested = home_cache
    res = vdg.clean_home_hf_stale_revisions(
        apply=False, max_age_days=7.0, cache_root=cache_root, now=now
    )
    assert res.skipped is False
    # delete_revisions called with EXACTLY the one stale hash.
    assert requested == [("old_unrefd",)]
    # Every keep rule survives: newest-but-old (keep-newest), old+ref'd
    # (keep-ref'd), young+unref'd (age gate), single-revision (newest by
    # construction), repo-b newest.
    for kept in ("newest_old", "old_refd", "b_young", "b_newest", "single_old"):
        assert all(kept not in req for req in requested), kept
    assert any("org/repo-a" in d for d in res.detail)  # per-repo detail line
    # No STALE line names repo-b (nothing reapable there). The unified tier's
    # attribution arm (#1376) names EVERY repo in its own detail lines, so the
    # original blanket "repo-b appears nowhere" assertion is narrowed to the
    # stale-revision line format.
    assert not any("stale revision" in d and "org/repo-b" in d for d in res.detail)


# ─── 2: report-only vs apply (+ sidecar row) ─────────────────────────────────


def test_home_hf_tier_report_only_vs_apply(home_cache, repo):
    cache_root, now, executed, _requested = home_cache

    res = vdg.clean_home_hf_stale_revisions(
        apply=False, max_age_days=7.0, cache_root=cache_root, now=now
    )
    assert res.bytes_freed == 55555  # expected_freed_size reported
    assert res.total_discovered_bytes == 55555
    assert executed == []  # dry-run executes NOTHING
    assert _read_sidecar(repo) == []  # report-only persists NO sidecar row
    assert any("would trim" in d for d in res.detail)

    res2 = vdg.clean_home_hf_stale_revisions(
        apply=True, max_age_days=7.0, cache_root=cache_root, now=now
    )
    assert executed == [("old_unrefd",)]  # strategy.execute() called once
    assert res2.bytes_freed == 55555
    rows = _read_sidecar(repo)
    assert len(rows) == 1
    assert rows[0]["kind"] == "home-hf-revisions-trimmed"
    assert rows[0]["repos"] == {"org/repo-a": 1}  # per-repo stale counts
    assert rows[0]["bytes"] == 55555
    assert any("trimmed 1 revision(s)" in d for d in res2.detail)


# ─── 3: age gate — env parsing + a real parameter ────────────────────────────


def test_home_hf_tier_age_gate_env(home_cache, monkeypatch):
    cache_root, now, _executed, requested = home_cache

    # Env honored; blank/invalid/negative -> the 7.0 default.
    monkeypatch.setenv("EPS_VM_HOME_HF_REVISION_MAX_AGE_DAYS", "3.5")
    assert vdg.home_hf_revision_max_age_days() == 3.5
    monkeypatch.setenv("EPS_VM_HOME_HF_REVISION_MAX_AGE_DAYS", "")
    assert vdg.home_hf_revision_max_age_days() == 7.0
    monkeypatch.setenv("EPS_VM_HOME_HF_REVISION_MAX_AGE_DAYS", "bogus")
    assert vdg.home_hf_revision_max_age_days() == 7.0
    monkeypatch.setenv("EPS_VM_HOME_HF_REVISION_MAX_AGE_DAYS", "-1")
    assert vdg.home_hf_revision_max_age_days() == 7.0

    # And the cache-root env: blank -> ~ default; set -> expanded override.
    monkeypatch.setenv("EPS_VM_HOME_HF_CACHE", "")
    assert vdg.home_hf_cache_root() == Path("~/.cache/huggingface").expanduser()
    monkeypatch.setenv("EPS_VM_HOME_HF_CACHE", str(cache_root))
    assert vdg.home_hf_cache_root() == cache_root

    # max_age_days=0.0 targets the young unref'd revision too — the gate is a
    # real parameter, not a constant (newest + ref'd + single still kept).
    res = vdg.clean_home_hf_stale_revisions(
        apply=False, max_age_days=0.0, cache_root=cache_root, now=now
    )
    assert res.skipped is False
    assert len(requested) == 1
    assert sorted(requested[0]) == ["b_young", "old_unrefd"]
    for kept in ("newest_old", "old_refd", "b_newest", "single_old"):
        assert kept not in requested[0], kept


# ─── 4: missing hub cache -> clean skip ──────────────────────────────────────


def test_home_hf_tier_missing_cache_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(vdg, "_running_pod_side", lambda: False)
    res = vdg.clean_home_hf_stale_revisions(apply=True, cache_root=tmp_path)  # no hub/
    assert res.skipped and "no hub cache" in res.skip_reason


# ─── 5: pod guard refuses before the scan ────────────────────────────────────


def test_home_hf_tier_pod_guard(tmp_path, monkeypatch):
    (tmp_path / "hub").mkdir()
    monkeypatch.setattr(vdg, "_running_pod_side", lambda: True)

    def _never(hub):
        raise AssertionError("_scan_hf_cache must not be reached under the pod guard")

    monkeypatch.setattr(vdg, "_scan_hf_cache", _never)
    res = vdg.clean_home_hf_stale_revisions(apply=True, cache_root=tmp_path)
    assert res.skipped and "pod-side" in res.skip_reason


# ─── 6: every failure degrades to skipped ────────────────────────────────────


def test_home_hf_tier_failure_degrades_to_skipped(tmp_path, monkeypatch):
    (tmp_path / "hub").mkdir()
    monkeypatch.setattr(vdg, "_running_pod_side", lambda: False)

    def _boom(hub):
        raise ImportError("huggingface_hub gone")

    monkeypatch.setattr(vdg, "_scan_hf_cache", _boom)
    res = vdg.clean_home_hf_stale_revisions(apply=True, cache_root=tmp_path)
    assert res.skipped and "ImportError" in res.skip_reason

    # An execute()-time failure degrades the same way; bytes_freed stays 0.
    now = time.time()

    def _bad_info(hub):
        info = _fake_home_cache_info(now, [], [])

        def delete_revisions(*hashes):
            return SimpleNamespace(
                expected_freed_size=1,
                execute=lambda: (_ for _ in ()).throw(OSError("corrupt cache")),
            )

        info.delete_revisions = delete_revisions
        return info

    monkeypatch.setattr(vdg, "_scan_hf_cache", _bad_info)
    res2 = vdg.clean_home_hf_stale_revisions(apply=True, cache_root=tmp_path, now=now)
    assert res2.skipped and "OSError" in res2.skip_reason
    assert res2.bytes_freed == 0  # set only AFTER a successful execute()


# ─── 7: nothing stale -> clean (not skipped) no-op ───────────────────────────


def test_home_hf_tier_no_stale_is_clean_noop(home_cache):
    cache_root, now, executed, requested = home_cache
    # 365d gate: every unref'd non-newest revision is younger than the cutoff
    # (matches the live-cache trace at plan time: 0 targets today).
    res = vdg.clean_home_hf_stale_revisions(
        apply=True, max_age_days=365.0, cache_root=cache_root, now=now
    )
    assert res.skipped is False
    assert res.bytes_freed == 0
    assert executed == [] and requested == []
    assert any("no unref'd revision" in d for d in res.detail)


# ─── 8: run_guard registration rides the tier (d) opt-in ─────────────────────


def test_run_guard_home_hf_tier_rides_optin(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("tier (e) must not run without reclaim_tiers + explicit tmp_root")

    calls: list = []

    def _stub_tiers():
        monkeypatch.setattr(vdg, "clean_uv_cache", lambda apply: vdg.TierResult(name="uv-cache"))
        monkeypatch.setattr(
            vdg,
            "clean_terminal_download_caches",
            lambda apply, **k: vdg.TierResult(name="terminal-download-caches"),
        )
        monkeypatch.setattr(
            vdg, "clean_stale_logs", lambda *a, **k: vdg.TierResult(name="stale-logs")
        )
        monkeypatch.setattr(
            vdg,
            "clean_vm_workspace_hf_cache",
            lambda apply, **k: vdg.TierResult(name="workspace-hf-cache"),
        )

    def _patch_disk():
        state = {"calls": 0}

        def fake_used(path="/"):
            state["calls"] += 1
            return 90.0 if state["calls"] == 1 else 40.0

        monkeypatch.setattr(vdg, "disk_used_pct", fake_used)
        monkeypatch.setattr(vdg, "disk_free_gb", lambda path="/": 50.0)

    _stub_tiers()

    # (a) tmp_root=None (the library shape) never reaches tier (e).
    monkeypatch.setattr(vdg, "clean_home_hf_stale_revisions", _boom)
    _patch_disk()
    res = vdg.run_guard(apply=True, threshold=85.0, data_root=tmp_path / "data")
    assert res.triggered is True

    # (b) the data-disk shape (reclaim_tiers=False, tmp_root=None) never does.
    _patch_disk()
    res2 = vdg.run_guard(
        apply=True,
        threshold=85.0,
        data_root=tmp_path / "data",
        reclaim_tiers=False,
        tmp_root=None,
    )
    assert res2.triggered is True

    # (c) with an explicit tmp_root the tier IS called exactly once.
    def _recorder(apply, **k):
        calls.append(apply)
        return vdg.TierResult(name="home-hf-revisions")

    monkeypatch.setattr(vdg, "clean_home_hf_stale_revisions", _recorder)
    _patch_disk()
    res3 = vdg.run_guard(
        apply=True, threshold=85.0, data_root=tmp_path / "data", tmp_root=tmp_path / "faketmp"
    )
    assert res3.triggered is True
    assert calls == [True]
    assert "home-hf-revisions" in {t.name for t in res3.tiers}
