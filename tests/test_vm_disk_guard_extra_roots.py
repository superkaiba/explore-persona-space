"""Offline unit tests for the #2096 multi-root tier (e) of the VM disk guard
(scripts/vm_disk_guard.py): the relocated ``HF_HUB_CACHE`` hub cache on the
data disk (the #1369 relocation) gets the same revision-reap + size-cap
coverage as the home root, triggered from the DATA-DISK pass.

Covers (plan #2096 §6):
  T1  run_guard(reclaim_tiers=False, hf_cache_roots=[root]) over threshold →
      tier (e) runs on the extra root, reaps the unreferenced-stale revision,
      keep contract holds (newest + ref'd survive).
  T2  same call minus hf_cache_roots → NO tier-(e) result (hermeticity — the
      #911 data-disk no-sweep pin stays true by construction).
  T3  an extra root resolving to the HOME root's hub is skipped ("boot-disk
      pass owns it"); duplicate/symlinked roots in the list are deduped.
  T4  EPS_VM_EXTRA_HF_CACHE_ROOTS env semantics (unset → user-templated
      default; set-but-empty → []; colon list) + EPS_VM_DATA_HF_CACHE_CAP_GB
      fail-soft parsing.
  T5  source-scan pin: run_guard never calls extra_hf_cache_roots() itself
      (the resolver lives ONLY in main() — the #911 production_tmp_root
      pattern).
  T6  main()-level with a real-mount monkeypatch: the data-disk JSON payload
      carries the extra-root tier result.
  T7  a missing extra root is a clean skipped no-op.
  T8  escalation state keys + ack sentinels are namespaced per root; the
      default-root key stays byte-identical to today's, and a home-root ack
      can never suppress the data-root alert.

Fixtures mirror tests/test_vm_disk_guard_home_hf_cache.py (fake HFCacheInfo
via SimpleNamespace) + tests/test_vm_disk_guard_data_disk.py (_patch_disk,
main()-level real-mount monkeypatch). Loaded via importlib exactly like
tests/test_vm_disk_guard.py.
"""

import ast
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

NOW = 1_700_000_000.0
DAY = 86400.0
HOUR = 3600.0


# ─── fixtures (mirroring test_vm_disk_guard_home_hf_cache.py) ────────────────


def _mk_file(blob_path: str, atime: float):
    return SimpleNamespace(blob_path=blob_path, blob_last_accessed=atime)


def _mk_rev(commit: str, *, refs=(), last_modified: float, files=(), size: int = 1_000):
    return SimpleNamespace(
        commit_hash=commit,
        refs=frozenset(refs),
        last_modified=last_modified,
        files=list(files),
        size_on_disk=size,
    )


def _mk_repo(repo_id: str, *, repo_type="model", last_accessed: float, revisions, size=None):
    revisions = list(revisions)
    if size is None:
        size = sum(r.size_on_disk for r in revisions)
    return SimpleNamespace(
        repo_id=repo_id,
        repo_type=repo_type,
        size_on_disk=size,
        last_accessed=last_accessed,
        revisions=revisions,
    )


def _mk_info(repos, executed: list, *, expected_freed: int = 12_345, warnings=()):
    def delete_revisions(*hashes):
        return SimpleNamespace(
            expected_freed_size=expected_freed,
            requested=hashes,
            execute=lambda: executed.append(tuple(hashes)),
        )

    return SimpleNamespace(
        repos=list(repos), warnings=list(warnings), delete_revisions=delete_revisions
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Hermetic multi-root tier-(e) rig: repo_root -> tmp_path (ack sentinels
    + any state path land in the sandbox), pod guard off, recorders on the
    sidecar + Telegram writers, ambient env knobs cleared, and a fixture
    EXTRA cache root (parent dir holding hub/)."""
    monkeypatch.setattr(vdg, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(ced, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(vdg, "_running_pod_side", lambda: False)
    for knob in (
        "EPS_VM_EXTRA_HF_CACHE_ROOTS",
        "EPS_VM_DATA_HF_CACHE_CAP_GB",
        "EPS_VM_HOME_HF_CACHE",
        "EPS_VM_HOME_HF_CACHE_CAP_GB",
        "EPS_VM_HOME_HF_SIZE_CAP_MIN_AGE_HOURS",
    ):
        monkeypatch.delenv(knob, raising=False)
    events: list[tuple[dict, bool]] = []
    pushes: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        vdg, "append_disk_guard_event", lambda ev, *, apply=True: events.append((ev, apply))
    )
    monkeypatch.setattr(
        ced, "append_disk_guard_event", lambda ev, *, apply=True: events.append((ev, apply))
    )
    monkeypatch.setattr(
        vdg, "_telegram_push", lambda msg, apply: pushes.append((msg, apply)) or True
    )
    extra_root = tmp_path / "eps-data" / "huggingface-cache"
    (extra_root / "hub").mkdir(parents=True)
    data_root = tmp_path / "wt-data"
    data_root.mkdir()
    return SimpleNamespace(
        tmp_path=tmp_path,
        extra_root=extra_root,
        data_root=data_root,
        events=events,
        pushes=pushes,
    )


def _patch_disk(monkeypatch, before_pct, after_pct, free_gb=200.0):
    """disk_used_pct: before_pct on the 1st call, after_pct after (run_guard
    reads pre + post)."""
    state = {"calls": 0}

    def fake_used(path="/"):
        state["calls"] += 1
        return before_pct if state["calls"] == 1 else after_pct

    monkeypatch.setattr(vdg, "disk_used_pct", fake_used)
    monkeypatch.setattr(vdg, "disk_free_gb", lambda path="/": free_gb)


def _fresh_repo_with_stale_rev():
    """A FRESH multi-revision repo (arm 2 territory): newest+ref'd kept, one
    unreferenced-stale-cold candidate. Sizes stay KB-scale — far under the
    40 GB escalate threshold, so the tier's lazy production-state load can
    never fire from run_guard's state=None path (plan critic note (c))."""
    return _mk_repo(
        "org/data",
        repo_type="dataset",
        last_accessed=NOW - 1 * HOUR,
        revisions=[
            _mk_rev(
                "mainref1",
                refs={"main"},
                last_modified=NOW - 1 * DAY,
                files=[_mk_file("blobM", NOW - 1 * DAY)],
            ),
            _mk_rev(
                "cold2222",
                last_modified=NOW - 20 * DAY,
                files=[_mk_file("blobC", NOW - 20 * DAY)],
            ),
        ],
    )


# ─── T1: the data-disk pass reaps the extra root through tier (e) ────────────


def test_extra_root_tier_reaps_in_data_disk_pass(env, monkeypatch):
    executed: list = []
    info = _mk_info([_fresh_repo_with_stale_rev()], executed, expected_freed=4242)
    monkeypatch.setattr(vdg, "_scan_hf_cache", lambda hub: info)
    _patch_disk(monkeypatch, before_pct=96.0, after_pct=40.0)

    res = vdg.run_guard(
        apply=True,
        threshold=85.0,
        data_root=env.data_root,
        disk_path="/mnt/eps-data",
        reclaim_tiers=False,
        now=NOW,
        hf_cache_roots=[env.extra_root],
    )
    assert res.triggered is True
    # Ordering: tier (b) then tier (e) — the /-rooted uv/log tiers never ran.
    assert [t.name for t in res.tiers] == ["terminal-download-caches", "home-hf-revisions"]
    tier = res.tiers[-1]
    assert tier.skipped is False
    # The unreferenced-stale revision was reaped; newest + ref'd survive.
    assert executed == [("cold2222",)]
    assert tier.bytes_freed == 4242


# ─── T2: no opt-in → no tier-(e) result (hermeticity) ────────────────────────


def test_extra_root_absent_without_optin(env, monkeypatch):
    scanned = {"n": 0}

    def _scan(hub):
        scanned["n"] += 1
        return _mk_info([], [])

    monkeypatch.setattr(vdg, "_scan_hf_cache", _scan)
    _patch_disk(monkeypatch, before_pct=96.0, after_pct=40.0)

    res = vdg.run_guard(
        apply=True,
        threshold=85.0,
        data_root=env.data_root,
        disk_path="/mnt/eps-data",
        reclaim_tiers=False,
        now=NOW,
    )
    assert res.triggered is True
    assert {t.name for t in res.tiers} == {"terminal-download-caches"}
    assert scanned["n"] == 0  # no root opted in → no cache scan at all


# ─── T3: home-collision skip + resolved-root dedup ───────────────────────────


def test_extra_root_skips_home_collision(env, monkeypatch):
    # (a) an extra root that IS the home root: the boot-disk pass owns it.
    home_root = env.tmp_path / "home-cache"
    (home_root / "hub").mkdir(parents=True)
    monkeypatch.setattr(vdg, "home_hf_cache_root", lambda: home_root)
    res = vdg.clean_home_hf_stale_revisions(
        False, cache_root=home_root, now=NOW, state={}, root_tag="deadbeef"
    )
    assert res.skipped is True
    assert "boot-disk pass owns it" in res.skip_reason

    # (b) duplicate + symlink-aliased roots are deduped by resolved hub/
    # (first wins): exactly ONE tier-(e) result for the three entries.
    alias = env.tmp_path / "alias-cache"
    alias.symlink_to(env.extra_root, target_is_directory=True)
    monkeypatch.setattr(vdg, "_scan_hf_cache", lambda hub: _mk_info([], []))
    _patch_disk(monkeypatch, before_pct=96.0, after_pct=40.0)
    res2 = vdg.run_guard(
        apply=False,
        threshold=85.0,
        data_root=env.data_root,
        disk_path="/mnt/eps-data",
        reclaim_tiers=False,
        now=NOW,
        hf_cache_roots=[env.extra_root, env.extra_root, alias],
    )
    assert [t.name for t in res2.tiers].count("home-hf-revisions") == 1


# ─── T4: env semantics for the resolver + the data-disk cap ──────────────────


def test_extra_roots_env_semantics(monkeypatch):
    import getpass

    # Unset → the user-templated data-disk default.
    monkeypatch.delenv("EPS_VM_EXTRA_HF_CACHE_ROOTS", raising=False)
    assert vdg.extra_hf_cache_roots() == [
        Path(f"/mnt/eps-data/{getpass.getuser()}/huggingface-cache")
    ]
    # Set-but-empty → [] (the explicit kill switch).
    monkeypatch.setenv("EPS_VM_EXTRA_HF_CACHE_ROOTS", "")
    assert vdg.extra_hf_cache_roots() == []
    # Colon-separated list; blank/whitespace entries dropped.
    monkeypatch.setenv("EPS_VM_EXTRA_HF_CACHE_ROOTS", "/a/b::  :/c/d")
    assert vdg.extra_hf_cache_roots() == [Path("/a/b"), Path("/c/d")]

    # Cap getter: fail-soft parsing mirrors home_hf_cache_cap_bytes.
    monkeypatch.delenv("EPS_VM_DATA_HF_CACHE_CAP_GB", raising=False)
    assert vdg.data_hf_cache_cap_gb() == vdg.DEFAULT_DATA_HF_CACHE_CAP_GB == 150.0
    for bad in ("inf", "nan", "-5", "abc"):
        monkeypatch.setenv("EPS_VM_DATA_HF_CACHE_CAP_GB", bad)
        assert vdg.data_hf_cache_cap_gb() == 150.0
    monkeypatch.setenv("EPS_VM_DATA_HF_CACHE_CAP_GB", "200")
    assert vdg.data_hf_cache_cap_gb() == 200.0


# ─── T5: run_guard never resolves the extra roots itself (#911 mirror) ───────


class _ExtraRootsRefVisitor(ast.NodeVisitor):
    """Collects every Name/Attribute reference to ``extra_hf_cache_roots``
    with its enclosing-function stack (a FunctionDef's own name is a plain
    attribute, not a Name node, so the definition never self-reports) —
    the #911 ``production_tmp_root`` pin's shape."""

    def __init__(self):
        self.stack: list[str] = []
        self.refs: list[tuple[str, ...]] = []

    def visit_FunctionDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Name(self, node):
        if node.id == "extra_hf_cache_roots":
            self.refs.append(tuple(self.stack))

    def visit_Attribute(self, node):
        if node.attr == "extra_hf_cache_roots":
            self.refs.append(tuple(self.stack))
        self.generic_visit(node)


def test_run_guard_never_resolves_extra_roots_itself():
    visitor = _ExtraRootsRefVisitor()
    visitor.visit(ast.parse((_SCRIPTS / "vm_disk_guard.py").read_text()))
    assert visitor.refs, "expected at least the main() call site"
    for stack in visitor.refs:
        assert stack and stack[-1] == "main", (
            "vm_disk_guard.py: extra_hf_cache_roots referenced outside main(): "
            f"{' > '.join(stack) or '<module>'}"
        )


# ─── T6: main() wires the extra roots into the data-disk pass ────────────────


def test_main_data_disk_pass_passes_extra_roots(env, monkeypatch, capsys):
    import json as _json

    monkeypatch.setenv("EPS_VM_EXTRA_HF_CACHE_ROOTS", str(env.extra_root))
    # A fresh single-revision repo: nothing reapable, everything attributed.
    now = time.time()
    info = _mk_info(
        [
            _mk_repo(
                "org/model",
                last_accessed=now - 60.0,
                revisions=[_mk_rev("aaa111", refs={"main"}, last_modified=now - 120.0)],
            )
        ],
        [],
    )
    monkeypatch.setattr(vdg, "_scan_hf_cache", lambda hub: info)
    monkeypatch.setattr(vdg, "_is_mounted", lambda p: True)  # simulate a live mount
    # / healthy (boot pass NOT triggered → hermetic), data disk over threshold.
    monkeypatch.setattr(vdg, "disk_used_pct", lambda path="/": 40.0 if path == "/" else 96.0)
    monkeypatch.setattr(vdg, "disk_free_gb", lambda path="/": 200.0)

    rc = vdg.main(["--data-disk-path", "/mnt/eps-data", "--json", "--no-push"])
    payload = _json.loads(capsys.readouterr().out)
    # Boot pass under threshold → no tiers; data-disk pass carries tier (e).
    assert payload["triggered"] is False
    dd = payload["data_disk"]
    assert dd["triggered"] is True
    tier_names = [t["name"] for t in dd["tiers"]]
    assert "home-hf-revisions" in tier_names
    hf = next(t for t in dd["tiers"] if t["name"] == "home-hf-revisions")
    assert hf["skipped"] is False
    assert [r["repo"] for r in hf["hf_repo_attributions"]] == ["org/model"]
    assert rc == 2  # data disk still over threshold after the report-only pass


# ─── T7: a missing extra root is a clean skipped no-op ───────────────────────


def test_extra_root_missing_dir_noop(env, monkeypatch):
    missing = env.tmp_path / "not-there" / "huggingface-cache"
    monkeypatch.setattr(
        vdg, "_scan_hf_cache", lambda hub: (_ for _ in ()).throw(AssertionError("never scanned"))
    )
    _patch_disk(monkeypatch, before_pct=96.0, after_pct=40.0)
    res = vdg.run_guard(
        apply=True,
        threshold=85.0,
        data_root=env.data_root,
        disk_path="/mnt/eps-data",
        reclaim_tiers=False,
        now=NOW,
        hf_cache_roots=[missing],
    )
    tier = next(t for t in res.tiers if t.name == "home-hf-revisions")
    assert tier.skipped is True
    assert "no hub cache at" in tier.skip_reason
    assert tier.bytes_freed == 0


# ─── T8: escalation keys + ack sentinels are namespaced per root ─────────────


def _big_repo(size: int):
    return _mk_repo(
        "superkaiba1/explore-persona-space-data",
        repo_type="dataset",
        last_accessed=NOW - 1 * HOUR,
        revisions=[_mk_rev("aaa111", refs={"main"}, last_modified=NOW - 1 * DAY, size=size)],
        size=size,
    )


def test_escalation_key_namespaced_by_root(env, monkeypatch):
    repo_key = "dataset/superkaiba1/explore-persona-space-data"
    monkeypatch.setattr(vdg, "_scan_hf_cache", lambda hub: _mk_info([_big_repo(60 * 10**9)], []))

    def esc_events():
        return [ev for ev, _ in env.events if ev["kind"] == "home-hf-cache-repo-escalation"]

    # Default (home) root: root_tag="" → key + sentinel byte-identical to today.
    home_root = env.tmp_path / "home-cache"
    (home_root / "hub").mkdir(parents=True)
    state_home: dict = {}
    vdg.clean_home_hf_stale_revisions(True, cache_root=home_root, now=NOW, state=state_home)
    assert state_home == {f"hf:{repo_key}:50": 60 * 10**9}
    assert len(esc_events()) == 1
    assert "cache_root" not in esc_events()[0]  # default-root row: no new field
    assert "home HF hub cache" in env.pushes[0][0]

    # Extra (data-disk) root: tagged key, distinct sentinel, root named in
    # the push + the sidecar row's additive cache_root field.
    tag = vdg._hf_root_tag(env.extra_root)
    assert tag and tag != ""
    default_ack = vdg._hf_ack_sentinel_path(repo_key, 50.0)
    tagged_ack = vdg._hf_ack_sentinel_path(repo_key, 50.0, tag)
    assert default_ack != tagged_ack
    # A HOME-root ack must NOT suppress the data-root alert (#2096 D6).
    default_ack.parent.mkdir(parents=True, exist_ok=True)
    default_ack.touch()
    state_extra: dict = {}
    vdg.clean_home_hf_stale_revisions(
        True, cache_root=env.extra_root, now=NOW, state=state_extra, root_tag=tag
    )
    assert state_extra == {f"hf:{tag}:{repo_key}:50": 60 * 10**9}
    assert len(esc_events()) == 2  # fired despite the home-root ack
    assert esc_events()[1]["cache_root"] == str(env.extra_root)
    assert f"HF hub cache at {env.extra_root}" in env.pushes[1][0]
    # The TAGGED ack sentinel does suppress the tagged root.
    tagged_ack.touch()
    vdg.clean_home_hf_stale_revisions(
        True, cache_root=env.extra_root, now=NOW, state=dict(state_extra), root_tag=tag
    )
    assert len(esc_events()) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
