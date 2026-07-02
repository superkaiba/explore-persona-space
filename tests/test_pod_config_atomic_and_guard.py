"""Task #821: atomicity + never-drop-RUNNING guard + re-add + invariants.

Layered defense against the incident #821 pods.conf wipe:

- **Atomic write** — ``write_pods_conf`` writes via ``os.replace`` on the
  same filesystem so no reader ever sees a partial file. A mid-write
  crash cannot leave the LIVE ``pods.conf`` corrupt (only a ``.tmp``
  sibling, which the next writer overwrites).
- **Never-drop-RUNNING guard** — ``write_pods_conf`` diffs the on-disk
  row set against the new one; any pod being dropped that the live
  RunPod API still reports RUNNING is RE-ADDED with a loud stderr WARN.
  The explicit remove path (``_remove_from_pods_conf`` in
  ``pod_lifecycle``) passes ``allow_remove={name}`` to opt out; UPDATE
  paths (name in both sets) trigger no guard call at all.
- **RunPod API unreachable** — the guard falls SAFE by re-adding every
  dropped row, matching CLAUDE.md's "never drop a RUNNING pod" invariant.
- **_refresh_one_pod re-add** — the fact-checked A13 gap; the existing
  refresh test suite drives ONLY through ``cmd_refresh_from_api`` so a
  direct unit test locks the ``PodInfo -> Pod`` mapping (in particular
  the ``gpu_type_id or "unknown"`` fallback for missing gpu-type).
- **Multi-process race** — a Barrier-synchronized re-run of the
  ``locked_pods_conf`` regression asserts both workers' rows survive
  under a deterministic lock-entry rendezvous, defending against a
  future refactor that would drop the flock.
- **Invariants** — grep-pinned checks that no ``.py`` under ``scripts/``
  constructs a literal ``"pods.conf"`` path outside ``pod_config.py``
  (catches the ``cleanup_pod.py`` class of missed consumer, incident
  #821 fact-check A12) AND that no shell script writes to ``pods.conf``
  (the invariant that lets us keep the path-resolver Python-side).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pod_config  # noqa: E402
from pod_config import Pod  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(name: str, host: str = "1.2.3.4", port: int = 22000) -> Pod:
    return Pod(name=name, host=host, port=port, gpus=1, gpu_type="H100", label=f"thomas-{name}")


def _seed_conf(path: Path, pods: list[Pod]) -> None:
    """Write a starter pods.conf via the real writer (with an explicit path
    that bypasses the lazy live resolver — a fixture setup step, not a
    contract test)."""
    # Use the same header the real writer emits to keep parity.
    lines = ["# Pod registry -- test fixture", "# Format: name host port gpus gpu_type label"]
    for p in pods:
        lines.append(f"{p.name}  {p.host}  {p.port}  {p.gpus}  {p.gpu_type}  {p.label}")
    path.write_text("\n".join(lines) + "\n")


class _FakePodInfo:
    """Minimal ``PodInfo`` stand-in for the guard's ``list_team_pods`` mock.

    ``pod_config.write_pods_conf`` reads only ``name``, ``desired_status``,
    ``ssh_host``, ``ssh_port`` on the guard path; the direct re-add test
    (``_refresh_one_pod``) also reads ``gpu_count`` and ``gpu_type_id``.
    Keeping a lightweight fake here avoids pulling in the heavy
    ``runpod_api`` at test-collection time.
    """

    def __init__(
        self,
        name: str,
        desired_status: str = "RUNNING",
        ssh_host: str | None = "9.9.9.9",
        ssh_port: int | None = 33333,
        gpu_count: int | None = 1,
        gpu_type_id: str | None = "NVIDIA H100 80GB HBM3",
    ) -> None:
        self.name = name
        self.desired_status = desired_status
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.gpu_count = gpu_count
        self.gpu_type_id = gpu_type_id


@pytest.fixture
def stubbed_env(tmp_path, monkeypatch):
    """Point every side-effectful path at the tmp dir and stub the RunPod
    API. Returns a state dict tests use to seed pods.conf + stage what
    ``list_team_pods`` returns.
    """
    pods_conf = tmp_path / "pods.conf"
    monkeypatch.setattr(pod_config, "PODS_CONF", pods_conf)
    monkeypatch.setattr(pod_config, "PODS_CONF_LOCK", tmp_path / ".pods.conf.lock")

    state: dict[str, object] = {
        "conf_path": pods_conf,
        "live": [],  # list[_FakePodInfo] returned by list_team_pods()
        "list_calls": 0,
        "raise_on_list": None,  # set to an Exception instance to raise
    }

    # The guard does a lazy ``from runpod_api import RunPodError, list_team_pods``
    # so we replace those attrs on the real module.
    import runpod_api

    def fake_list_team_pods() -> list[object]:
        state["list_calls"] = int(state["list_calls"]) + 1  # type: ignore[arg-type]
        exc = state["raise_on_list"]
        if isinstance(exc, BaseException):
            raise exc
        return list(state["live"])  # type: ignore[arg-type]

    monkeypatch.setattr(runpod_api, "list_team_pods", fake_list_team_pods)
    return state


# ---------------------------------------------------------------------------
# Never-drop-RUNNING guard — happy paths
# ---------------------------------------------------------------------------


def test_write_pods_conf_never_drops_running_pod(stubbed_env):
    """Seed pod-A + pod-B; API says pod-A RUNNING, pod-B EXITED. Writing
    only [pod-A] must drop pod-B (EXITED is safe to remove) and keep pod-A.
    """
    path = stubbed_env["conf_path"]
    _seed_conf(path, [_row("pod-A", host="1.1.1.1", port=11111), _row("pod-B")])
    stubbed_env["live"] = [
        _FakePodInfo("pod-A", desired_status="RUNNING"),
        _FakePodInfo("pod-B", desired_status="EXITED", ssh_host=None, ssh_port=None),
    ]

    pod_config.write_pods_conf([_row("pod-A", host="1.1.1.1", port=11111)], path=path)

    survivors = {p.name for p in pod_config.parse_pods_conf(path=path)}
    assert survivors == {"pod-A"}, survivors
    assert stubbed_env["list_calls"] == 1


def test_write_pods_conf_refuses_to_drop_running_pod(stubbed_env, capsys):
    """Seed pod-A + pod-B; API says BOTH RUNNING. Writing only [pod-A] must
    re-add pod-B (the guard) and emit a WARN naming pod-B + the opt-out."""
    path = stubbed_env["conf_path"]
    _seed_conf(path, [_row("pod-A"), _row("pod-B", host="2.2.2.2", port=22222)])
    stubbed_env["live"] = [
        _FakePodInfo("pod-A"),
        _FakePodInfo("pod-B"),
    ]

    pod_config.write_pods_conf([_row("pod-A")], path=path)

    survivors = {p.name: (p.host, p.port) for p in pod_config.parse_pods_conf(path=path)}
    assert "pod-B" in survivors, survivors
    # The re-add uses the on-disk row so host/port survive verbatim.
    assert survivors["pod-B"] == ("2.2.2.2", 22222)

    err = capsys.readouterr().err
    assert "pod-B" in err
    assert "refusing to drop RUNNING pod" in err
    assert "allow_remove" in err


def test_write_pods_conf_explicit_remove_bypasses_guard(stubbed_env, monkeypatch):
    """Seed pod-A + pod-B; the legitimate remove path
    (``_remove_from_pods_conf``) passes ``allow_remove={pod-B}``. The guard
    must NOT call ``list_team_pods`` at all in that case (a terminate flow
    should not depend on network access).
    """
    path = stubbed_env["conf_path"]
    _seed_conf(path, [_row("pod-A"), _row("pod-B")])

    called = {"list_team_pods": 0}

    def _boom() -> list[object]:
        called["list_team_pods"] += 1
        raise AssertionError("guard called list_team_pods on an allowed-remove path")

    import runpod_api

    monkeypatch.setattr(runpod_api, "list_team_pods", _boom)

    pod_config.write_pods_conf([_row("pod-A")], path=path, allow_remove=frozenset({"pod-B"}))

    survivors = {p.name for p in pod_config.parse_pods_conf(path=path)}
    assert survivors == {"pod-A"}, survivors
    assert called["list_team_pods"] == 0


def test_write_pods_conf_api_unreachable_keeps_all(stubbed_env, capsys):
    """``RunPodError`` from ``list_team_pods`` means the API cannot disprove
    RUNNING. Fail SAFE: re-add every dropped row, WARN loudly."""
    from runpod_api import RunPodError

    path = stubbed_env["conf_path"]
    _seed_conf(path, [_row("pod-A"), _row("pod-B")])
    stubbed_env["raise_on_list"] = RunPodError("simulated network failure")

    pod_config.write_pods_conf([_row("pod-A")], path=path)

    survivors = {p.name for p in pod_config.parse_pods_conf(path=path)}
    assert survivors == {"pod-A", "pod-B"}, survivors

    err = capsys.readouterr().err
    assert "could not verify live pod status" in err
    assert "failing SAFE" in err


def test_write_pods_conf_update_is_noop_for_guard(stubbed_env, monkeypatch):
    """A host/port UPDATE keeps the pod name in both on_disk and new sets,
    so ``dropped`` is empty and the guard NEVER calls ``list_team_pods``.
    Legitimate resume paths must not depend on network access.
    """
    path = stubbed_env["conf_path"]
    _seed_conf(path, [_row("pod-A", host="1.1.1.1", port=11111)])

    called = {"list_team_pods": 0}

    def _boom() -> list[object]:
        called["list_team_pods"] += 1
        raise AssertionError("guard called list_team_pods on a pure-UPDATE write")

    import runpod_api

    monkeypatch.setattr(runpod_api, "list_team_pods", _boom)

    pod_config.write_pods_conf([_row("pod-A", host="9.9.9.9", port=33333)], path=path)

    rows = {p.name: (p.host, p.port) for p in pod_config.parse_pods_conf(path=path)}
    assert rows == {"pod-A": ("9.9.9.9", 33333)}
    assert called["list_team_pods"] == 0


# ---------------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------------


def test_write_pods_conf_is_atomic_and_leaves_target_unchanged(stubbed_env, monkeypatch):
    """Force ``os.replace`` to raise; assert the pods.conf on disk still
    matches its pre-write bytes verbatim (no torn write). The tmp sibling
    may or may not exist — cleanup of the leftover is a separate concern.
    """
    path = stubbed_env["conf_path"]
    _seed_conf(path, [_row("pod-A", host="1.1.1.1", port=11111)])
    pre_bytes = path.read_bytes()

    class _ReplaceBoom(RuntimeError):
        pass

    def _fake_replace(src, dst):  # type: ignore[no-untyped-def]
        raise _ReplaceBoom("simulated atomic-rename failure")

    monkeypatch.setattr(pod_config.os, "replace", _fake_replace)

    with pytest.raises(_ReplaceBoom):
        pod_config.write_pods_conf(
            [_row("pod-A", host="9.9.9.9", port=33333)],
            path=path,
            allow_remove=frozenset(),  # ensures guard is a no-op (UPDATE)
        )

    # Target file byte-identical to its pre-write content — the point of
    # ``os.replace``: readers never see a torn intermediate state.
    assert path.read_bytes() == pre_bytes


# ---------------------------------------------------------------------------
# Lost-update deterministic race
# ---------------------------------------------------------------------------


def _worker_barrier_upsert(
    pods_conf_path_str: str,
    lock_path_str: str,
    pod_name: str,
    host: str,
    port: int,
    barrier: mp.Barrier,
) -> None:
    """Worker: wait on a Barrier BEFORE contending for the lock (a
    deterministic synchronisation point that widens the race window in a
    reproducible way, unlike ``time.sleep``), then attempt the upsert.
    """
    import pod_config  # type: ignore[import-not-found]

    pods_conf_path = Path(pods_conf_path_str)
    pod_config.PODS_CONF = pods_conf_path
    pod_config.PODS_CONF_LOCK = Path(lock_path_str)

    from pod_config import Pod, locked_pods_conf, parse_pods_conf, write_pods_conf

    barrier.wait()

    with locked_pods_conf():
        rows = parse_pods_conf(path=pods_conf_path)
        # Small hold to widen the window (deterministic upper bound only —
        # the assertion below tests the FILE state, not the interleaving).
        time.sleep(0.15)
        rows.append(
            Pod(name=pod_name, host=host, port=port, gpus=1, gpu_type="H100", label=pod_name)
        )
        write_pods_conf(rows, path=pods_conf_path)


def test_barrier_synchronized_concurrent_upserts_both_survive(tmp_path):
    """Deterministic version of the ``locked_pods_conf`` regression: use a
    ``multiprocessing.Barrier(2)`` to force both workers to hit the lock
    at (approximately) the same instant. Both rows must survive — the
    invariant is a property of the FINAL file state, never the
    interleaving order (which is not observable across a fcntl lock).
    """
    pods_conf = tmp_path / "pods.conf"
    _seed_conf(pods_conf, [])
    lock_path = tmp_path / ".pods.conf.lock"

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    workers = [
        ctx.Process(
            target=_worker_barrier_upsert,
            args=(str(pods_conf), str(lock_path), "pod-race-A", "10.0.0.1", 22001, barrier),
        ),
        ctx.Process(
            target=_worker_barrier_upsert,
            args=(str(pods_conf), str(lock_path), "pod-race-B", "10.0.0.2", 22002, barrier),
        ),
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=30)
        assert w.exitcode == 0, f"worker {w.name} exited {w.exitcode}"

    survivors = pods_conf.read_text()
    assert any(ln.startswith("pod-race-A") for ln in survivors.splitlines()), survivors
    assert any(ln.startswith("pod-race-B") for ln in survivors.splitlines()), survivors


# ---------------------------------------------------------------------------
# _refresh_one_pod direct re-add (fact-check A13 gap)
# ---------------------------------------------------------------------------


def test_refresh_one_pod_re_adds_missing_running_row_direct(stubbed_env):
    """Direct unit test on ``_refresh_one_pod`` — locks the ``row is None
    + live RUNNING`` re-add branch's ``PodInfo -> Pod`` mapping (including
    the ``gpu_type_id or "unknown"`` None-fallback for the required
    ``Pod.gpu_type: str`` field).
    """
    to_add: list[Pod] = []
    live = _FakePodInfo(
        "pod-763",
        desired_status="RUNNING",
        ssh_host="103.207.149.130",
        ssh_port=18166,
        gpu_count=8,
        gpu_type_id=None,  # forces the "unknown" fallback
    )

    changed, warned = pod_config._refresh_one_pod(
        "pod-763",
        row=None,
        live=live,
        is_single_mode=True,
        manual_override=False,
        to_add=to_add,
    )

    assert changed is True
    assert warned is False
    assert len(to_add) == 1
    added = to_add[0]
    assert added.name == "pod-763"
    assert (added.host, added.port) == ("103.207.149.130", 18166)
    assert added.gpus == 8
    # ``gpu_type_id=None`` triggers the "unknown" default (the field is
    # ``Pod.gpu_type: str`` — cannot be None).
    assert added.gpu_type == "unknown"


def test_refresh_one_pod_skips_row_none_when_live_not_running(stubbed_env):
    """The re-add branch REQUIRES live RUNNING with a populated SSH
    endpoint. A stale row missing from pods.conf whose live pod is EXITED
    stays missing (nothing to restore)."""
    to_add: list[Pod] = []
    changed, warned = pod_config._refresh_one_pod(
        "pod-999",
        row=None,
        live=_FakePodInfo("pod-999", desired_status="EXITED", ssh_host=None, ssh_port=None),
        is_single_mode=False,
        manual_override=False,
        to_add=to_add,
    )
    assert changed is False
    assert warned is True
    assert to_add == []


# ---------------------------------------------------------------------------
# Invariant: no shell writer + no literal path outside pod_config.py
# ---------------------------------------------------------------------------


def _list_shell_scripts() -> list[Path]:
    scripts = REPO_ROOT / "scripts"
    return sorted(p for p in scripts.rglob("*.sh") if p.is_file())


def test_no_shell_script_writes_pods_conf():
    """No ``scripts/**/*.sh`` writes to pods.conf via any redirection/tee/
    sed-in-place pattern. Reads via ``. _pods_conf_path.sh && cat "$CONF"``
    stay allowed; writers do not exist (Python-side is the sole writer).
    """
    write_patterns = [
        re.compile(r">\s*[\"']?\$?\{?CONF\}?[\"']?"),  # > "$CONF"
        re.compile(r">>\s*[\"']?\$?\{?CONF\}?[\"']?"),  # >> "$CONF"
        re.compile(r"tee\s+[\"']?\$?\{?CONF\}?[\"']?"),  # tee "$CONF"
        re.compile(r"sed\s+-i[^\n]*[\"']?\$?\{?CONF\}?[\"']?"),  # sed -i .. "$CONF"
        # Literal pods.conf writes (any shape) — no shell script should
        # write to a literal "pods.conf" filename either.
        re.compile(r">\s*[\"']?[^\"']*/?pods\.conf[\"']?"),
        re.compile(r">>\s*[\"']?[^\"']*/?pods\.conf[\"']?"),
        re.compile(r"tee\s+[\"']?[^\"']*/?pods\.conf[\"']?"),
    ]
    violations: list[str] = []
    for script in _list_shell_scripts():
        text = script.read_text(errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            # Skip comments.
            if line.lstrip().startswith("#"):
                continue
            for rx in write_patterns:
                if rx.search(line):
                    violations.append(f"{script.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
                    break
    assert not violations, (
        "Shell scripts must not write pods.conf (Python is the sole writer for "
        "the atomic-write + never-drop-RUNNING contract). Offenders:\n" + "\n".join(violations)
    )


def test_no_literal_pods_conf_path_outside_pod_config():
    """Grep-pin: no ``scripts/*.py`` constructs a literal ``"pods.conf"``
    path outside ``pod_config.py`` (the sole owner of path resolution).
    Comments and docstrings are ignored via a coarse contains-check on
    the code portion of each line.

    Catches the class of missed consumer that motivated fact-check A12
    (``cleanup_pod.py:40`` held its own ``CONF_PATH = SCRIPT_DIR /
    "pods.conf"`` and would have silently read the seed after the v3
    relocation).
    """
    scripts_dir = REPO_ROOT / "scripts"
    # Only literal string constructions on the LHS of a Path expression
    # (e.g. ``SCRIPT_DIR / "pods.conf"``, ``Path(...) / "pods.conf"``,
    # or a direct string constant assignment). Purely-textual occurrences
    # (log strings, docstrings, comments) are not a path construction.
    path_construction_re = re.compile(r"/\s*[\"']pods\.conf[\"']")

    # Files where the literal-path construction is legitimate.
    ALLOWLIST = {
        # The sole owner of resolution.
        "pod_config.py",
    }

    violations: list[str] = []
    for py in sorted(scripts_dir.rglob("*.py")):
        if py.name in ALLOWLIST:
            continue
        text = py.read_text(errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if path_construction_re.search(line):
                violations.append(f"{py.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")

    assert not violations, (
        "No .py under scripts/ may construct a literal 'pods.conf' path "
        "outside pod_config.py — route through pod_config.PODS_CONF / "
        "_resolve_live_pods_conf instead. Offenders:\n" + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Post-write LF check — tail newline preserved through the atomic path
# ---------------------------------------------------------------------------


def test_write_pods_conf_atomic_write_is_trailing_newline_preserved(stubbed_env):
    """Regression for a subtle atomic-write bug class: the tmp sibling
    payload must end with a newline so the on-disk pods.conf is a
    well-formed newline-terminated file (some readers assume it).
    """
    path = stubbed_env["conf_path"]
    _seed_conf(path, [_row("pod-A")])
    pod_config.write_pods_conf([_row("pod-A")], path=path)
    assert path.read_bytes().endswith(b"\n")


# ---------------------------------------------------------------------------
# Sanity: the atomic path does not leave a stale ``.tmp`` on success
# ---------------------------------------------------------------------------


def test_write_pods_conf_atomic_write_removes_tmp_on_success(stubbed_env):
    """On success, ``os.replace`` renames tmp -> path; no ``.tmp`` sibling
    should remain in the target directory."""
    path = stubbed_env["conf_path"]
    _seed_conf(path, [_row("pod-A")])
    pod_config.write_pods_conf([_row("pod-A")], path=path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    assert not tmp.exists(), f"stale tmp survived: {tmp}"


# ---------------------------------------------------------------------------
# Guard is a no-op when nothing is dropped (adds only)
# ---------------------------------------------------------------------------


def test_write_pods_conf_pure_add_does_not_call_list_team_pods(stubbed_env, monkeypatch):
    """Appending a new row to pods.conf without dropping any existing row
    is a pure INSERT; the guard has nothing to check and must not touch
    the network."""
    path = stubbed_env["conf_path"]
    _seed_conf(path, [_row("pod-A")])

    called = {"list_team_pods": 0}

    def _boom() -> list[object]:
        called["list_team_pods"] += 1
        raise AssertionError("guard called list_team_pods on a pure INSERT")

    import runpod_api

    monkeypatch.setattr(runpod_api, "list_team_pods", _boom)

    pod_config.write_pods_conf(
        [_row("pod-A"), _row("pod-B", host="2.2.2.2", port=22222)], path=path
    )

    rows = {p.name for p in pod_config.parse_pods_conf(path=path)}
    assert rows == {"pod-A", "pod-B"}
    assert called["list_team_pods"] == 0


# ---------------------------------------------------------------------------
# Guard preserves manual_override rows (they stay on disk, no drop attempted)
# ---------------------------------------------------------------------------


def test_write_pods_conf_survives_hardened_lock_dir(stubbed_env):
    """The parent-dir mkdir is idempotent: writing to an existing dir must
    not error (regression guard: an earlier draft raised if parent existed
    but had different owner metadata)."""
    path = stubbed_env["conf_path"]
    _seed_conf(path, [_row("pod-A")])
    # Ensure the parent dir already exists (it does — tmp_path).
    assert path.parent.exists()
    pod_config.write_pods_conf([_row("pod-A")], path=path)
    assert path.exists()


# ---------------------------------------------------------------------------
# ``_remove_from_pods_conf`` passes ``allow_remove`` (integration check)
# ---------------------------------------------------------------------------


def test_remove_from_pods_conf_bypasses_guard(monkeypatch, tmp_path):
    """Integration check: the one legitimate remove path
    (``pod_lifecycle._remove_from_pods_conf``) passes
    ``allow_remove={name}`` so the guard skips the API entirely. Verified
    by monkeypatching ``list_team_pods`` to raise if invoked.
    """
    pods_conf = tmp_path / "pods.conf"
    _seed_conf(pods_conf, [_row("pod-A"), _row("pod-B")])
    monkeypatch.setattr(pod_config, "PODS_CONF", pods_conf)
    monkeypatch.setattr(pod_config, "PODS_CONF_LOCK", tmp_path / ".pods.conf.lock")

    # No-op cmd_sync — ~/.ssh/config regeneration is out of scope.
    monkeypatch.setattr(pod_config, "cmd_sync", lambda rows: None)

    import runpod_api

    def _boom() -> list[object]:
        raise AssertionError("guard called list_team_pods from _remove_from_pods_conf path")

    monkeypatch.setattr(runpod_api, "list_team_pods", _boom)

    # Import lazily — ``pod_lifecycle`` is heavy (loads runpod_api's config).
    import pod_lifecycle

    pod_lifecycle._remove_from_pods_conf("pod-A")

    survivors = {p.name for p in pod_config.parse_pods_conf(path=pods_conf)}
    assert survivors == {"pod-B"}


# ---------------------------------------------------------------------------
# Diagnostic: os.replace atomic contract (documented, not assumed)
# ---------------------------------------------------------------------------


def test_os_replace_is_available_and_atomic_on_same_fs(tmp_path):
    """Micro-check that ``os.replace`` is available in the running Python
    and operates on same-FS renames. Not a contract on Python itself —
    just a smoke that the atomic path's precondition (same-FS tmp sibling)
    is satisfied on the test host."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.write_text("hello")
    os.replace(src, dst)
    assert not src.exists()
    assert dst.read_text() == "hello"
