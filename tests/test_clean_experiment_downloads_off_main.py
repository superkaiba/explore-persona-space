"""Tests for the off-main graceful degrade in
``scripts/clean_experiment_downloads.py`` (task #924).

On a NON-``main`` remote checkout (the GCE/pod ``issue-<N>`` clone lanes),
``task_workflow.repo_root()`` raises its branch-guard ``RuntimeError`` (the
#841 att-6 crash: a ``--depth 1 --branch`` clone with no local ``main``) or
silently routes to an empty-``data/`` pin worktree. The fix routes the four
RESOLUTION sites (``_data_root`` / ``_worktree_data_roots`` / the sidecar /
``_rel_name``) through ``_resolution_root()`` (checkout-local off-main), and
gate 1 (#773 active-consumer) becomes ATTEMPT-AND-CATCH: the read is still
attempted; only a ``RuntimeError`` from it is caught (loud WARN + sidecar row
``kind: "off-main-consumer-gate-skipped"`` + empty protected set). A
succeeding read (full clone with a local ``main``) keeps the gate ON.

The script lives under ``scripts/`` (not an importable package), so it is
loaded via importlib exactly like
``tests/test_clean_experiment_downloads_pod_side_short_circuit.py``.
"""

import importlib.util
import json
import subprocess
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

import explore_persona_space.task_workflow as tw  # noqa: E402  (after _load, deliberate)

# The #841 att-6 branch-guard message, VERBATIM (task_workflow.py
# _ensure_managed_main_worktree) — any resolution site still reaching
# repo_root() in the off-main fixture fails the test loudly with it.
_BRANCH_GUARD_MSG = (
    "primary checkout /workspace/eps-issue-841 is on 'issue-841' and has no local `main` "
    "branch to route task.py writes through; create `main` (or check it out) before "
    "running task.py."
)


def _raise_branch_guard():
    raise RuntimeError(_BRANCH_GUARD_MSG)


# ─── fixtures / helpers ──────────────────────────────────────────────────────


@pytest.fixture
def fake_off_main_clone(tmp_path, monkeypatch):
    """Simulate the GCE ``--depth 1 --branch issue-841`` clone shape.

    The probe inputs are monkeypatched (primary checkout = ``tmp_path``,
    attached branch ``issue-841``, NOT the shared VM via the real
    ``EPS_SHARED_VM=0`` override path), while BOTH ``ced.repo_root`` and
    ``task_workflow.repo_root`` RAISE the #841 branch-guard error verbatim —
    so the four resolution sites are proven to never reach ``repo_root()``,
    and the gate-1 read (``tasks_dir()`` -> ``repo_root()``) exercises the
    attempt-and-catch. Caches + a sibling ``store/`` + ``notes.txt`` live
    under ``tmp_path/data/issue_841/``."""
    issue_dir = tmp_path / "data" / "issue_841"
    for cache in ("hf_dl", "g1_dl"):
        d = issue_dir / cache
        d.mkdir(parents=True)
        (d / "blob.bin").write_bytes(b"x" * 2048)
    store = issue_dir / "store"
    store.mkdir()
    (store / "keep.pt").write_bytes(b"y" * 4096)
    (issue_dir / "notes.txt").write_text("keep me")

    monkeypatch.setattr(ced, "_running_pod_side", lambda: False)
    monkeypatch.setattr(ced, "primary_checkout_root", lambda: tmp_path)
    monkeypatch.setattr(ced, "_checkout_branch", lambda p: "issue-841")
    monkeypatch.setenv("EPS_SHARED_VM", "0")  # exercise the REAL is_shared_vm_env override
    monkeypatch.setattr(ced, "repo_root", _raise_branch_guard)
    monkeypatch.setattr(tw, "repo_root", _raise_branch_guard)

    ced._off_main_checkout_root.cache_clear()
    yield tmp_path
    ced._off_main_checkout_root.cache_clear()


def _read_sidecar(root: Path) -> list[dict]:
    path = root / ".claude" / "cache" / "disk-guard-events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# ─── test 1: hero — the #841 att-6 scenario end to end via main() ────────────


def test_off_main_incremental_apply_reaps_checkout_local_caches(fake_off_main_clone, capsys):
    """``main(["841", "--incremental", "--apply"])`` on the simulated clone:
    rc 0, hf_dl + g1_dl reaped from the CHECKOUT-LOCAL data/, store/ +
    notes.txt intact, WARN names the checkout + #924, sidecar row written.
    The raising repo_root monkeypatch proves the four resolution sites never
    call it AND exercises the gate-1 catch (the read IS attempted, raises the
    #841 error verbatim, and is caught at the gate-1 site only)."""
    clone = fake_off_main_clone
    rc = ced.main(["841", "--incremental", "--apply"])
    assert rc == 0
    issue_dir = clone / "data" / "issue_841"
    assert not (issue_dir / "hf_dl").exists()
    assert not (issue_dir / "g1_dl").exists()
    assert (issue_dir / "store" / "keep.pt").is_file()
    assert (issue_dir / "notes.txt").is_file()
    err = capsys.readouterr().err
    assert "non-main checkout" in err
    assert "#924" in err
    rows = [r for r in _read_sidecar(clone) if r["kind"] == "off-main-consumer-gate-skipped"]
    assert len(rows) == 1
    assert rows[0]["task"] == 841
    assert rows[0]["checkout"] == str(clone)


# ─── test 2: dry-run deletes nothing ─────────────────────────────────────────


def test_off_main_dry_run_deletes_nothing(fake_off_main_clone, capsys):
    """No ``--apply``: rc 0, both cache dirs still on disk, would-remove
    report lines present."""
    clone = fake_off_main_clone
    rc = ced.main(["841", "--incremental"])
    assert rc == 0
    issue_dir = clone / "data" / "issue_841"
    assert (issue_dir / "hf_dl").is_dir()
    assert (issue_dir / "g1_dl").is_dir()
    out = capsys.readouterr().out
    assert "would remove" in out
    assert "hf_dl" in out and "g1_dl" in out


# ─── test 3: gate-1 read is ATTEMPTED; only RuntimeError is caught ───────────


def test_off_main_gate_read_attempted_and_caught(fake_off_main_clone, monkeypatch, capsys):
    """Arm (a): a RuntimeError from the gate read is caught — the reap
    succeeds with the WARN + sidecar row, proving the read WAS attempted."""
    clone = fake_off_main_clone
    attempted = {"n": 0}

    def _gate_raises(issue_n):
        attempted["n"] += 1
        raise RuntimeError(_BRANCH_GUARD_MSG)

    monkeypatch.setattr(ced, "_active_consumer_protected_issues", _gate_raises)
    res = ced.clean_issue_downloads(841, apply=True)
    assert attempted["n"] == 1  # the read was attempted, not probe-keyed-skipped
    assert sorted(res.removed) == ["data/issue_841/g1_dl", "data/issue_841/hf_dl"]
    assert res.skipped == [] and res.failed == []
    assert "non-main checkout" in capsys.readouterr().err
    assert any(r["kind"] == "off-main-consumer-gate-skipped" for r in _read_sidecar(clone))


def test_off_main_gate_read_non_runtimeerror_propagates(fake_off_main_clone, monkeypatch):
    """Arm (b): a non-RuntimeError from the gate read still PROPAGATES —
    the catch is narrow, not a blanket swallow."""

    def _gate_asserts(issue_n):
        raise AssertionError("not a branch-guard failure")

    monkeypatch.setattr(ced, "_active_consumer_protected_issues", _gate_asserts)
    with pytest.raises(AssertionError, match="not a branch-guard failure"):
        ced.clean_issue_downloads(841, apply=True)


# ─── test 3-bis: full-clone shape — a SUCCEEDING gate read stays ON ──────────


def test_off_main_full_clone_gate_stays_on(fake_off_main_clone, monkeypatch):
    """Probe non-None but the gate read SUCCEEDS (the full-clone / pin-worktree
    topology) returning a live consumer: the #773 protection is ENFORCED —
    both cache dirs are SKIPPED with the consumer reason and stay on disk."""
    clone = fake_off_main_clone
    monkeypatch.setattr(ced, "_active_consumer_protected_issues", lambda n: {841: [999]})
    res = ced.clean_issue_downloads(841, apply=True)
    issue_dir = clone / "data" / "issue_841"
    assert (issue_dir / "hf_dl").is_dir()
    assert (issue_dir / "g1_dl").is_dir()
    assert res.removed == []
    skipped_names = sorted(name for name, _ in res.skipped)
    assert skipped_names == ["data/issue_841/g1_dl", "data/issue_841/hf_dl"]
    for _, reason in res.skipped:
        assert "#999" in reason and "data/issue_841/" in reason
    rows = [r for r in _read_sidecar(clone) if r["kind"] == "active-consumer-reap-skipped"]
    assert len(rows) == 2
    # And no gate-skip row: the gate was ON, never skipped.
    assert not any(r["kind"] == "off-main-consumer-gate-skipped" for r in _read_sidecar(clone))


# ─── test 4: nested-store parity gate (#679) still binds off-main ────────────


def test_off_main_nested_store_parity_gate_still_binds(fake_off_main_clone, monkeypatch):
    """An unmirrored nested store/ under hf_dl blocks its reap off-main
    exactly as on-main (fail-toward-keep); g1_dl is still reaped."""
    clone = fake_off_main_clone
    nested = clone / "data" / "issue_841" / "hf_dl" / "store"
    nested.mkdir()
    (nested / "gen.pt").write_bytes(b"z" * 1024)
    monkeypatch.setattr(ced, "_hf_file_sizes", lambda repo, revision="main": None)

    res = ced.clean_issue_downloads(841, apply=True)
    assert (clone / "data" / "issue_841" / "hf_dl").is_dir()
    assert not (clone / "data" / "issue_841" / "g1_dl").exists()
    assert [name for name, _ in res.skipped] == ["data/issue_841/hf_dl"]
    assert res.removed == ["data/issue_841/g1_dl"]


# ─── tests 5-7: probe classification ─────────────────────────────────────────


def test_probe_none_on_main_branch(fake_off_main_clone, monkeypatch):
    """Primary attached on `main` => probe None (the normal VM path)."""
    monkeypatch.setattr(ced, "_checkout_branch", lambda p: "main")
    ced._off_main_checkout_root.cache_clear()
    assert ced._off_main_checkout_root() is None


def test_probe_none_on_detached_or_unresolvable(fake_off_main_clone, monkeypatch):
    """Detached HEAD (branch None) => None; an unresolvable layout
    (primary_checkout_root raising RuntimeError) => None. Either way
    repo_root() keeps raising its own loud, unchanged error downstream."""
    monkeypatch.setattr(ced, "_checkout_branch", lambda p: None)
    ced._off_main_checkout_root.cache_clear()
    assert ced._off_main_checkout_root() is None

    def _unresolvable():
        raise RuntimeError("resolved repo root has no `tasks/` directory")

    monkeypatch.setattr(ced, "primary_checkout_root", _unresolvable)
    ced._off_main_checkout_root.cache_clear()
    assert ced._off_main_checkout_root() is None
    ced._off_main_checkout_root.cache_clear()


def test_probe_refuses_on_shared_vm(fake_off_main_clone, monkeypatch, capsys):
    """EPS_SHARED_VM=1 (overriding the fixture's "0") + branch issue-841:
    the probe REFUSES the degrade (None + a SHARED VM warning), and the
    pathological off-main-VM state keeps today's loud crash on the gate."""
    monkeypatch.setenv("EPS_SHARED_VM", "1")
    ced._off_main_checkout_root.cache_clear()  # WARN prints on the first uncached probe
    assert ced._off_main_checkout_root() is None
    assert "SHARED VM" in capsys.readouterr().err

    # Probe None => gate 1 is a plain call; the raising repo_root propagates.
    with pytest.raises(RuntimeError, match="has no local"):
        ced.clean_issue_downloads(841, apply=False)


# ─── test 8: on-main path still calls the consumer gate (VM invariant) ───────


def test_on_main_path_still_calls_consumer_gate(tmp_path, monkeypatch):
    """Probe pinned None (on-main): the consumer gate is computed exactly once
    via a sentinel — the VM path is unchanged by #924."""
    issue_dir = tmp_path / "data" / "issue_841"
    (issue_dir / "hf_dl").mkdir(parents=True)
    (issue_dir / "hf_dl" / "blob.bin").write_bytes(b"x" * 512)
    monkeypatch.setattr(ced, "_off_main_checkout_root", lambda: None)
    monkeypatch.setattr(ced, "repo_root", lambda: tmp_path)
    calls = {"n": 0}

    def _sentinel(issue_n):
        calls["n"] += 1
        return {}

    monkeypatch.setattr(ced, "_active_consumer_protected_issues", _sentinel)
    res = ced.clean_issue_downloads(841, apply=False)
    assert calls["n"] == 1
    assert res.removed == ["data/issue_841/hf_dl"]


# ─── test 9: real-git shallow-branch-clone probe integration ─────────────────


def _git(*args, cwd=None):
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def test_probe_real_git_shallow_branch_clone(tmp_path, monkeypatch):
    """Integration (CPU seconds, no network): a REAL ``git clone --depth 1
    --branch issue-841`` of a tiny origin holding a ``tasks/`` tree; the
    UNmonkeypatched probe — resolver anchored at the clone via
    ``task_workflow._MODULE_DIR`` — classifies it as an off-main checkout and
    returns the clone root (closing the A2/A10 emulation-fidelity gap)."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git("init", "-q", "-b", "issue-841", cwd=origin)
    (origin / "tasks").mkdir()
    (origin / "tasks" / "REGISTRY.json").write_text("{}")
    _git("add", "tasks", cwd=origin)
    _git(
        "-c",
        "user.email=test@test",
        "-c",
        "user.name=test",
        "commit",
        "-q",
        "-m",
        "init",
        cwd=origin,
    )
    clone = tmp_path / "clone"
    _git("clone", "-q", "--depth", "1", "--branch", "issue-841", f"file://{origin}", str(clone))

    # Real _checkout_branch against the real clone (always asserted).
    assert ced._checkout_branch(clone) == "issue-841"

    # Anchor the REAL resolver at the clone (the probe's primary_checkout_root
    # resolves from task_workflow._MODULE_DIR) and run the unmonkeypatched
    # probe end to end.
    monkeypatch.setenv("EPS_SHARED_VM", "0")
    monkeypatch.setattr(tw, "_MODULE_DIR", clone)
    tw.invalidate_cache()
    ced._off_main_checkout_root.cache_clear()
    try:
        result = ced._off_main_checkout_root()
        assert result is not None
        assert Path(result).resolve() == clone.resolve()
    finally:
        # Never leak the clone-anchored resolution into other tests.
        ced._off_main_checkout_root.cache_clear()
        tw.invalidate_cache()
