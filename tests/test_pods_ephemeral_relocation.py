"""Task #1183: the live ``pods_ephemeral.json`` lives at
``<git-common-dir>/eps/pods_ephemeral.json`` — OUT of the git working tree —
mirroring the #821 pods.conf relocation.

Pins, per the approved plan §5:

- Resolver contract: git-common-dir resolution, seed→live migration
  (byte-identical, fast path on the second call), bootstrap when neither
  seed nor live exists, monkeypatch honor for BOTH public symbols
  (``pod_config.PODS_EPHEMERAL_JSON`` + ``pod_lifecycle.EPHEMERAL_STATE``),
  read-only-FS loud-WARN seed fallback.
- Writer contract: atomic write (crash injection leaves the original
  intact + ``.tmp`` cleaned), structural never-drop guard (re-add + WARN;
  ``allow_remove`` opt-out), ``_save_state``'s reconcile still drops
  API-validated entries without tripping the guard.
- The headline properties the task exists to deliver: the live file
  survives every destructive working-tree git op, and a
  barrier-synchronized two-process RMW loses neither update (the §1
  lost-update leg — the never-drop guard heals entry-level drops but NOT
  field-level stale overwrites, so this is the only pin on concurrency).
- Grep-pin: no quoted ``"pods_ephemeral.json"`` literal in ``scripts/*.py``
  outside ``pod_config.py`` (the sole owner of path resolution).

Test-isolation invariant (plan §4, HARD): every fixture uses tmp git repos
or monkeypatched seed constants + a tmp lock path — this suite must NEVER
touch the real ``<repo>/.git/eps/`` or the real ``scripts/.pods.conf.lock``.
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pod_config  # noqa: E402
import pod_lifecycle  # noqa: E402
from pod_lifecycle import EphemeralMetadata, EphemeralPod  # noqa: E402
from runpod_api import PodInfo  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers + fixtures
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


def _meta(name: str, *, issue: int, **overrides) -> EphemeralMetadata:
    base = {
        "name": name,
        "pod_id": f"pod-{name}",
        "issue": issue,
        "gpu_intent": "lora-7b",
        "ttl_days": 7,
        "stopped_at": None,
        "notes": "",
    }
    base.update(overrides)
    return EphemeralMetadata(**base)


def _info(name: str) -> PodInfo:
    return PodInfo(
        pod_id=f"pod-{name}",
        name=name,
        desired_status="RUNNING",
        gpu_count=1,
        gpu_type_id="NVIDIA H100 80GB HBM3",
        ssh_host="1.2.3.4",
        ssh_port=12345,
        created_at="2026-07-01T00:00:00Z",
    )


def _seed_payload() -> dict:
    """A 2-pod seed payload in the version-2 metadata-only schema."""
    return {
        "version": 2,
        "updated_at": "2026-07-01T00:00:00+00:00",
        "pods": {
            "pod-1": {
                "name": "pod-1",
                "pod_id": "pod-pod-1",
                "issue": 1,
                "gpu_intent": "lora-7b",
                "ttl_days": 7,
                "stopped_at": None,
                "notes": "",
                "manual_override": False,
                "extra": {},
            },
            "pod-2": {
                "name": "pod-2",
                "pod_id": "pod-pod-2",
                "issue": 2,
                "gpu_intent": "eval",
                "ttl_days": 7,
                "stopped_at": None,
                "notes": "",
                "manual_override": False,
                "extra": {},
            },
        },
    }


@pytest.fixture
def scratch_repo(tmp_path):
    """Minimal scratch git repo mirroring the real layout:

    <tmp>/repo/
        scripts/pods_ephemeral.json   # committed seed
        .git/                         # live file lands under .git/eps/
    """
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    _git(repo.parent, "init", str(repo))
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "config", "commit.gpgsign", "false")

    seed = repo / "scripts" / "pods_ephemeral.json"
    seed.write_text(json.dumps(_seed_payload(), indent=2) + "\n")
    _git(repo, "add", "scripts/pods_ephemeral.json")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "seed sidecar")
    return repo


def _point_resolver(repo: Path, monkeypatch) -> Path:
    """Point pod_config's module globals at ``repo`` with the seed==override
    shape so the resolver's live-vs-seed branch fires as in production —
    against the scratch repo, never the real checkout / lockfile."""
    seed = repo / "scripts" / "pods_ephemeral.json"
    monkeypatch.setattr(pod_config, "SCRIPT_DIR", repo / "scripts")
    monkeypatch.setattr(pod_config, "PODS_EPHEMERAL_SEED", seed)
    monkeypatch.setattr(pod_config, "PODS_EPHEMERAL_JSON", seed)
    monkeypatch.setattr(pod_config, "PODS_CONF_LOCK", repo / "scripts" / ".pods.conf.lock")
    return seed


# ---------------------------------------------------------------------------
# Resolver contract
# ---------------------------------------------------------------------------


def test_resolves_to_git_common_dir_eps(scratch_repo, monkeypatch):
    """When the LIVE file exists, the resolver returns
    ``<git-common-dir>/eps/pods_ephemeral.json`` — a path OUTSIDE the
    working tree."""
    _point_resolver(scratch_repo, monkeypatch)
    live_dir = scratch_repo / ".git" / "eps"
    live_dir.mkdir(parents=True)
    live = live_dir / "pods_ephemeral.json"
    live.write_text('{"version": 2, "pods": {}}\n')

    resolved = pod_config.resolve_live_pods_ephemeral()
    assert resolved == live
    assert resolved.relative_to(scratch_repo).parts[0] == ".git"


def test_migrates_seed_content_byte_identical_on_first_use(scratch_repo, monkeypatch):
    """First use copies the seed byte-for-byte to the live path (seed
    preserved); the second call takes the steady-state fast path — the
    locked migration section is never re-entered."""
    seed = _point_resolver(scratch_repo, monkeypatch)
    pre_seed_bytes = seed.read_bytes()

    resolved = pod_config.resolve_live_pods_ephemeral()
    assert resolved != seed
    assert resolved.exists()
    assert resolved.read_bytes() == pre_seed_bytes
    assert seed.read_bytes() == pre_seed_bytes

    def _no_lock():  # pragma: no cover - only fires on regression
        raise AssertionError(
            "second resolve must take the live fast path (no lock, no re-migration)"
        )

    monkeypatch.setattr(pod_config, "locked_pods_conf", _no_lock)
    assert pod_config.resolve_live_pods_ephemeral() == resolved


def test_bootstrap_when_neither_seed_nor_live_exists(scratch_repo, monkeypatch):
    """Neither seed nor live → live bootstrapped as an empty version-2
    sidecar, and a pod_lifecycle read of it yields no entries."""
    seed = _point_resolver(scratch_repo, monkeypatch)
    seed.unlink()

    resolved = pod_config.resolve_live_pods_ephemeral()
    assert json.loads(resolved.read_text()) == {"version": 2, "pods": {}}

    monkeypatch.setattr(pod_lifecycle, "EPHEMERAL_STATE", resolved)
    assert pod_lifecycle._read_metadata_file() == {}


def test_monkeypatched_pods_ephemeral_json_honored(tmp_path, monkeypatch):
    """A patched ``pod_config.PODS_EPHEMERAL_JSON`` is returned verbatim —
    no git resolution, no migration attempt."""
    target = tmp_path / "pods_ephemeral.json"
    monkeypatch.setattr(pod_config, "PODS_EPHEMERAL_JSON", target)

    def _boom():  # pragma: no cover - only fires on regression
        raise AssertionError("monkeypatched override must short-circuit before git resolution")

    monkeypatch.setattr(pod_config, "_git_common_dir", _boom)
    assert pod_config.resolve_live_pods_ephemeral() == target


def test_monkeypatched_ephemeral_state_honored(tmp_path, monkeypatch):
    """A patched ``pod_lifecycle.EPHEMERAL_STATE`` is honored by
    ``_resolve_state_path`` and by the read/write round-trip — the live
    resolver is never consulted."""
    state = tmp_path / "pods_ephemeral.json"
    monkeypatch.setattr(pod_lifecycle, "EPHEMERAL_STATE", state)

    def _boom():  # pragma: no cover - only fires on regression
        raise AssertionError("patched EPHEMERAL_STATE must never hit the live resolver")

    monkeypatch.setattr(pod_config, "resolve_live_pods_ephemeral", _boom)

    assert pod_lifecycle._resolve_state_path() == state
    pod_lifecycle._write_metadata_file({"pod-5": _meta("pod-5", issue=5)})
    assert state.exists()
    assert pod_lifecycle._read_metadata_file()["pod-5"].issue == 5


def test_read_only_fs_fallback_warns_and_returns_seed(scratch_repo, monkeypatch, capsys):
    """When the live dir cannot be created, the resolver WARNs loud and
    falls back to the seed. A FILE squatting at ``.git/eps`` makes the
    ``mkdir`` raise (``FileExistsError`` is an ``OSError``) — deterministic,
    no chmod/root caveats."""
    seed = _point_resolver(scratch_repo, monkeypatch)
    (scratch_repo / ".git" / "eps").write_text("not a dir")

    resolved = pod_config.resolve_live_pods_ephemeral()
    assert resolved == seed
    err = capsys.readouterr().err
    assert "WARN: cannot create live pods_ephemeral.json dir" in err
    assert "Falling back to seed" in err


# ---------------------------------------------------------------------------
# Writer contract: atomicity + structural never-drop guard
# ---------------------------------------------------------------------------


def test_write_metadata_file_atomic_on_replace_failure(tmp_path, monkeypatch):
    """Force ``os.replace`` to raise: the pre-existing sidecar stays
    byte-identical (no torn write), the ``.tmp`` sibling is cleaned, and
    the error surfaces (never swallowed)."""
    state = tmp_path / "pods_ephemeral.json"
    monkeypatch.setattr(pod_lifecycle, "EPHEMERAL_STATE", state)
    pod_lifecycle._write_metadata_file({"pod-1": _meta("pod-1", issue=1)})
    pre_bytes = state.read_bytes()

    class _ReplaceBoom(OSError):
        pass

    def _fake_replace(src, dst):  # type: ignore[no-untyped-def]
        raise _ReplaceBoom("simulated atomic-rename failure")

    monkeypatch.setattr(pod_config.os, "replace", _fake_replace)

    with pytest.raises(_ReplaceBoom):
        pod_lifecycle._write_metadata_file({"pod-1": _meta("pod-1", issue=1, notes="new")})

    assert state.read_bytes() == pre_bytes
    tmp = state.with_suffix(state.suffix + ".tmp")
    assert not tmp.exists(), f"stale tmp sibling {tmp} left behind after os.replace failure"


def test_never_drop_guard_readds_and_warns(tmp_path, monkeypatch, capsys):
    """An incoming dict silently missing an on-disk entry gets it RE-ADDED
    with a loud WARN naming the ``allow_remove`` opt-out."""
    state = tmp_path / "pods_ephemeral.json"
    monkeypatch.setattr(pod_lifecycle, "EPHEMERAL_STATE", state)
    pod_lifecycle._write_metadata_file(
        {"pod-1": _meta("pod-1", issue=1), "pod-2": _meta("pod-2", issue=2)}
    )
    capsys.readouterr()

    pod_lifecycle._write_metadata_file({"pod-1": _meta("pod-1", issue=1)})

    assert set(pod_lifecycle._read_metadata_file()) == {"pod-1", "pod-2"}
    err = capsys.readouterr().err
    assert "refusing to drop sidecar entry pod-2" in err
    assert "allow_remove" in err


def test_allow_remove_opt_out(tmp_path, monkeypatch, capsys):
    """A drop named in ``allow_remove`` goes through silently."""
    state = tmp_path / "pods_ephemeral.json"
    monkeypatch.setattr(pod_lifecycle, "EPHEMERAL_STATE", state)
    pod_lifecycle._write_metadata_file(
        {"pod-1": _meta("pod-1", issue=1), "pod-2": _meta("pod-2", issue=2)}
    )
    capsys.readouterr()

    pod_lifecycle._write_metadata_file({"pod-1": _meta("pod-1", issue=1)}, allow_remove={"pod-2"})

    assert set(pod_lifecycle._read_metadata_file()) == {"pod-1"}
    assert "refusing to drop" not in capsys.readouterr().err


def test_save_state_reconcile_still_drops_api_validated_entries(tmp_path, monkeypatch, capsys):
    """``_save_state`` over a ``_load_state`` view missing a dead pod drops
    it (its own ``allow_remove`` covers the reconcile) — the guard must not
    resurrect entries the live API already validated as gone."""
    state_file = tmp_path / "pods_ephemeral.json"
    monkeypatch.setattr(pod_lifecycle, "EPHEMERAL_STATE", state_file)
    pod_lifecycle._write_metadata_file(
        {"pod-1": _meta("pod-1", issue=1), "pod-2": _meta("pod-2", issue=2)}
    )
    capsys.readouterr()

    # The merged view is missing pod-2 (dropped by _load_state Branch 2:
    # present in the sidecar, absent from the live API).
    view = {"pod-1": EphemeralPod(metadata=_meta("pod-1", issue=1), info=_info("pod-1"))}
    pod_lifecycle._save_state(view)

    assert set(pod_lifecycle._read_metadata_file()) == {"pod-1"}
    assert "refusing to drop" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Writer-internal locking (§2 delta) — acquires on the unpatched path,
# skips under a monkeypatched symbol
# ---------------------------------------------------------------------------


def _recording_lock_factory(calls: dict):
    @contextlib.contextmanager
    def _recording_lock():
        calls["n"] += 1
        yield

    return _recording_lock


def test_writers_acquire_lock_on_unpatched_path(tmp_path, monkeypatch):
    """On the unpatched (production) shape — public symbol == seed symbol —
    both writers acquire ``locked_pods_conf``. The symbols are pointed at
    tmp files so the real sidecar/lock are never touched."""
    calls = {"n": 0}
    recording = _recording_lock_factory(calls)

    state = tmp_path / "pods_ephemeral.json"
    monkeypatch.setattr(pod_lifecycle, "EPHEMERAL_STATE", state)
    monkeypatch.setattr(pod_lifecycle, "_PODS_EPHEMERAL_JSON_MAIN", state)
    monkeypatch.setattr(pod_lifecycle, "locked_pods_conf", recording)
    monkeypatch.setattr(pod_config, "resolve_live_pods_ephemeral", lambda: state)

    pod_lifecycle._write_metadata_file({"pod-1": _meta("pod-1", issue=1)})
    assert calls["n"] == 1

    seed = tmp_path / "seed.json"
    monkeypatch.setattr(pod_config, "PODS_EPHEMERAL_SEED", seed)
    monkeypatch.setattr(pod_config, "PODS_EPHEMERAL_JSON", seed)
    monkeypatch.setattr(pod_config, "locked_pods_conf", recording)

    status = pod_config._set_manual_override("pod-1", value=True)
    assert calls["n"] == 2
    assert status is not None and "pod-1" in status


def test_writers_skip_lock_under_monkeypatched_symbols(tmp_path, monkeypatch):
    """With a patched ``EPHEMERAL_STATE`` / ``PODS_EPHEMERAL_JSON`` the
    writers take the nullcontext fast path — the (recording) lock is never
    acquired, so tests can never contend on the real shared lockfile."""
    calls = {"n": 0}
    recording = _recording_lock_factory(calls)
    monkeypatch.setattr(pod_lifecycle, "locked_pods_conf", recording)
    monkeypatch.setattr(pod_config, "locked_pods_conf", recording)

    state = tmp_path / "patched_state.json"
    monkeypatch.setattr(pod_lifecycle, "EPHEMERAL_STATE", state)  # != _PODS_EPHEMERAL_JSON_MAIN
    pod_lifecycle._write_metadata_file({"pod-9": _meta("pod-9", issue=9)})
    assert calls["n"] == 0

    patched_json = tmp_path / "patched.json"
    patched_json.write_text(
        json.dumps({"version": 2, "pods": {"pod-9": {"manual_override": False}}}) + "\n"
    )
    monkeypatch.setattr(pod_config, "PODS_EPHEMERAL_JSON", patched_json)  # != SEED
    status = pod_config._set_manual_override("pod-9", value=True)
    assert calls["n"] == 0
    assert status is not None
    assert json.loads(patched_json.read_text())["pods"]["pod-9"]["manual_override"] is True


# ---------------------------------------------------------------------------
# Headline 1: the live file survives every destructive git op
# ---------------------------------------------------------------------------


def test_live_json_survives_destructive_git_ops(scratch_repo, monkeypatch):
    """The task's raison d'être (mirror of
    ``test_pod_config_untracked_location.py::test_live_file_survives_
    destructive_git_ops``): after migration, every destructive working-tree
    git op leaves the LIVE file byte-intact while the dirtied seed is
    rewound to its committed state."""
    seed = _point_resolver(scratch_repo, monkeypatch)
    committed_seed = seed.read_bytes()

    live = pod_config.resolve_live_pods_ephemeral()
    payload = _seed_payload()
    payload["pods"]["pod-777"] = {
        "name": "pod-777",
        "pod_id": "pod-pod-777",
        "issue": 777,
        "gpu_intent": "eval",
        "ttl_days": 7,
        "stopped_at": None,
        "notes": "live-only entry (relocation test)",
        "manual_override": False,
        "extra": {},
    }
    live.write_text(json.dumps(payload, indent=2) + "\n")
    pre_live = live.read_bytes()

    # Dirty the seed so the rewind is observable.
    seed.write_text('{"version": 2, "pods": {}}\n')

    for cmd in (
        ["git", "reset", "--hard"],
        ["git", "checkout", "--", "."],
        ["git", "restore", "--", "."],
        ["git", "clean", "-fd"],
        ["git", "clean", "-fdx"],
    ):
        subprocess.run(cmd, cwd=scratch_repo, check=True, capture_output=True)
        assert live.exists(), (
            f"LIVE pods_ephemeral.json disappeared after {' '.join(cmd)} "
            f"(relocation contract broken)"
        )
        assert live.read_bytes() == pre_live, (
            f"LIVE pods_ephemeral.json mutated after {' '.join(cmd)} (relocation contract broken)"
        )

    assert seed.read_bytes() == committed_seed  # the seed WAS rewound — as designed


# ---------------------------------------------------------------------------
# Headline 2: barrier-synchronized two-process RMW — both updates survive
# ---------------------------------------------------------------------------


def _point_worker_at(repo: Path) -> None:
    """Worker-side global setup: seed == override at the scratch-repo paths,
    so the UNPATCHED/locked production path is what runs — against the tmp
    repo's lockfile, never the real one."""
    import pod_config as pc
    import pod_lifecycle as pl

    seed = repo / "scripts" / "pods_ephemeral.json"
    pc.SCRIPT_DIR = repo / "scripts"
    pc.PODS_EPHEMERAL_SEED = seed
    pc.PODS_EPHEMERAL_JSON = seed
    pc.PODS_CONF_LOCK = repo / "scripts" / ".pods.conf.lock"
    pl.EPHEMERAL_STATE = seed
    pl._PODS_EPHEMERAL_JSON_MAIN = seed


def _worker_notes_rmw(repo_str: str, barrier) -> None:
    """Field-level RMW through pod_lifecycle's locked path: set pod-1's
    notes, sleeping inside the critical section to widen the
    stale-overwrite window."""
    import pod_lifecycle as pl

    repo = Path(repo_str)
    _point_worker_at(repo)
    barrier.wait()
    with pl._metadata_lock():
        meta = pl._read_metadata_file()
        time.sleep(0.15)
        meta["pod-1"].notes = "A-note"
        pl._write_metadata_file(meta)


def _worker_override_rmw(repo_str: str, barrier) -> None:
    """Field-level RMW through pod_config's locked writer: flip pod-2's
    manual_override."""
    import pod_config as pc

    repo = Path(repo_str)
    _point_worker_at(repo)
    barrier.wait()
    status = pc._set_manual_override("pod-2", value=True)
    assert status is not None, "pod-2 missing from the sidecar in the worker"


def test_concurrent_metadata_rmw_both_survive(scratch_repo):
    """The §1 lost-update leg: two spawn-context processes RMW DIFFERENT
    FIELDS of different pod entries at (approximately) the same instant.
    Both updates must survive in the final live file — the never-drop guard
    heals entry-level drops but NOT field-level stale overwrites, so only
    the shared lock delivers this property."""
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    workers = [
        ctx.Process(target=_worker_notes_rmw, args=(str(scratch_repo), barrier)),
        ctx.Process(target=_worker_override_rmw, args=(str(scratch_repo), barrier)),
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=60)
        assert w.exitcode == 0, f"worker {w.name} exited {w.exitcode}"

    live = scratch_repo / ".git" / "eps" / "pods_ephemeral.json"
    assert live.exists(), "no worker migrated the seed to the live path"
    pods = json.loads(live.read_text())["pods"]
    assert pods["pod-1"]["notes"] == "A-note", pods
    assert pods["pod-2"]["manual_override"] is True, pods


# ---------------------------------------------------------------------------
# Grep-pin: pod_config.py is the sole owner of the literal path
# ---------------------------------------------------------------------------


def test_no_literal_pods_ephemeral_path_outside_pod_config():
    """No ``scripts/*.py`` may construct a bare quoted
    ``"pods_ephemeral.json"`` literal outside ``pod_config.py`` (the sole
    owner of path resolution) — the class of missed consumer that left
    ``pod.py`` reading a worktree-local seed pre-#1183. Broader than the
    pods.conf pin: catches ``os.path.join(...)`` / ``.joinpath(...)`` /
    bare-string constructions too (any both-sides-quoted literal)."""
    scripts_dir = REPO_ROOT / "scripts"
    literal_re = re.compile(r"[\"']pods_ephemeral\.json[\"']")
    allowlist = {"pod_config.py"}

    violations: list[str] = []
    for py in sorted(scripts_dir.rglob("*.py")):
        if py.name in allowlist:
            continue
        for lineno, line in enumerate(py.read_text(errors="replace").splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            if literal_re.search(line):
                violations.append(f"{py.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")

    assert not violations, (
        "No .py under scripts/ may construct a quoted 'pods_ephemeral.json' "
        "literal outside pod_config.py — route through "
        "pod_config.resolve_live_pods_ephemeral() / pod_lifecycle."
        "_resolve_state_path() instead. Offenders:\n" + "\n".join(violations)
    )
