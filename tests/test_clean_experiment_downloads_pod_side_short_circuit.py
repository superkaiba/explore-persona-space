"""Tests for the pod-side short-circuit in scripts/clean_experiment_downloads.py (#803).

On a RunPod pod, any use of task_workflow.repo_root() on a non-main HEAD auto-routes
to a managed worktree via git worktree add / git reset --hard, which hangs on
MooseFS-backed /workspace. main() must detect the pod and return 0 BEFORE any
helper calls repo_root(). Detector: /.dockerenv AND /workspace/logs both present.

The script lives under scripts/ (not an importable package), so it is loaded via
importlib exactly like tests/test_clean_experiment_downloads_parity.py.
"""

import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPTS / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


ced = _load("clean_experiment_downloads")


def test_pod_side_short_circuits_to_noop_zero(monkeypatch, capsys):
    """Detector True -> main() returns 0, prints the no-op notice, and never
    touches repo_root() (a repo_root that raises proves it was never called)."""
    monkeypatch.setattr(ced, "_running_pod_side", lambda: True)

    def _boom():
        raise AssertionError("repo_root() must not be called pod-side")

    monkeypatch.setattr(ced, "repo_root", _boom)
    rc = ced.main(["778", "--incremental", "--apply"])
    assert rc == 0
    assert "pod-side no-op" in capsys.readouterr().out


def test_detector_true_only_when_both_signals_present(monkeypatch):
    """_running_pod_side() is the AND of /.dockerenv and /workspace/logs."""
    real_exists = Path.exists
    real_is_dir = Path.is_dir

    def _exists(self):
        return str(self) == "/.dockerenv" or real_exists(self)

    def _is_dir(self):
        return str(self) == "/workspace/logs" or real_is_dir(self)

    # both present -> True
    monkeypatch.setattr(Path, "exists", _exists)
    monkeypatch.setattr(Path, "is_dir", _is_dir)
    assert ced._running_pod_side() is True

    # only /.dockerenv present (/workspace/logs missing) -> False.
    # NB /workspace/logs exists on this dev VM, so force it absent explicitly
    # rather than restoring the real is_dir (which would leave it present).
    def _is_dir_no_logs(self):
        return False if str(self) == "/workspace/logs" else real_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", _is_dir_no_logs)
    assert ced._running_pod_side() is False


def test_vm_side_runs_normal_dispatch(monkeypatch):
    """Detector False -> main() reaches the normal dispatch (proven by a
    sentinel cleaner that records the call and returns a real CleanResult)."""
    monkeypatch.setattr(ced, "_running_pod_side", lambda: False)
    called = {}

    def _fake_cleaner(issue, *, apply, data_root=None):
        called["issue"] = issue
        called["apply"] = apply
        return ced.CleanResult(issue_n=issue, apply=apply)

    monkeypatch.setattr(ced, "clean_issue_downloads_incremental", _fake_cleaner)
    rc = ced.main(["778", "--incremental", "--apply"])
    assert rc == 0
    assert called == {"issue": 778, "apply": True}
