"""#2606 — pod.py provision fails LOUD when its own preflight fails.

Pre-#2606, ``bootstrap_pod.sh`` step 10 swallowed the preflight rc with an
in-payload ``|| echo "PREFLIGHT-FAILED-AT-BOOTSTRAP rc=$?"``, so a broken pod
(venv/env not experiment-ready) exited 0 and printed BOOTSTRAP-OK. Three
surfaces changed, covered here + in the extended #1931 suite
(``tests/test_pod_lifecycle.py``, the ``#2606`` section):

* ``scripts/bootstrap_pod.sh`` step 10 — the payload's last line is the bare
  preflight command (rc propagates through ssh), the rc is captured locally
  (``PREFLIGHT_RC=$?``), and the failure branch prints the sentinel LOCALLY +
  exits ``EXIT_PREFLIGHT_FAILED=78``. Static pins + an EXECUTED shell-sandbox
  arm (stubbed ``ssh_cmd``) live in this file.
* ``scripts/pod_lifecycle.py`` — ``EXIT_BOOTSTRAP_PREFLIGHT_FAILED = 78`` +
  the kept-alive rc=78 branch (behavioral tests in test_pod_lifecycle.py;
  the 3-way constant parity pin lives here).
* ``src/explore_persona_space/backends/runpod.py`` — the launch-path rc-78
  disposition: best-effort exact-id terminate (#2038-family) + re-raise of
  the ORIGINAL relay error (tests here, mirroring the
  tests/test_issue2038_fallback_teardown.py patterns).

The shell sandbox executes the REAL step-10-through-EOF region (production
``set -euo pipefail`` regime, only ``ssh_cmd`` + log/step helpers stubbed) —
per the one-production-body-test rule, the presence pins alone cannot bind
the ``set +e`` rc-capture semantics.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

import pytest

from explore_persona_space.backends import runpod as RP
from explore_persona_space.backends.base import RunSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap_pod.sh"

_CONST_RE = re.compile(r"^EXIT_PREFLIGHT_FAILED=(\d+)$", re.MULTILINE)


def _text() -> str:
    return BOOTSTRAP.read_text(encoding="utf-8")


def _explode(*_a, **_k):
    raise AssertionError("seam must NOT be reached on this path (#2606)")


# ---------------------------------------------------------------------------
# Static pins (bootstrap_pod.sh) — Tests 5-7
# ---------------------------------------------------------------------------


def test_bootstrap_constant_present_and_old_swallow_gone() -> None:
    """(Test 5) the EXIT_PREFLIGHT_FAILED constant exists, the step-10 failure
    branch exits through it, and the old in-payload ``|| echo`` swallow is
    GONE from the whole file."""
    text = _text()
    m = _CONST_RE.search(text)
    assert m is not None, "EXIT_PREFLIGHT_FAILED assignment not found"
    assert 'exit "$EXIT_PREFLIGHT_FAILED"' in text
    assert '|| echo "PREFLIGHT-FAILED-AT-BOOTSTRAP' not in text, (
        "the pre-#2606 in-payload rc swallow is back — a failed preflight "
        "would exit 0 with BOOTSTRAP-OK again"
    )


def test_bootstrap_step10_ordering_failure_exit_before_summary() -> None:
    """(Test 7) static ordering pin: the step-10 fail-loud exit sits BEFORE
    the ``Bootstrap complete`` summary echo, so the exit structurally
    precedes any success banner (backs the sandbox arm's absent-banner
    assertion)."""
    text = _text()
    assert text.index('exit "$EXIT_PREFLIGHT_FAILED"') < text.index("Bootstrap complete for")


def test_exit_code_78_three_way_parity() -> None:
    """(Test 6) bootstrap_pod.sh / pod_lifecycle.py / backends/runpod.py all
    agree on 78, and 78 collides with none of the sibling structured exits
    (75 still-waiting, 76 stopped-pod collision, 77 CPU lane dry). The
    runpod.py copy is MIRRORED, never imported (imports stay base-only —
    the EXIT_STILL_WAITING precedent,
    tests/test_dispatch_issue_cli.py::test_exit_still_waiting_matches_pod_lifecycle)."""
    from scripts.pod_lifecycle import (
        EXIT_BOOTSTRAP_PREFLIGHT_FAILED as pl_code,
    )
    from scripts.pod_lifecycle import (
        EXIT_CPU_LANE_DRY,
    )

    m = _CONST_RE.search(_text())
    assert m is not None
    bash_code = int(m.group(1))
    assert bash_code == pl_code == RP.EXIT_BOOTSTRAP_PREFLIGHT_FAILED == 78
    assert 78 not in {
        RP.EXIT_STILL_WAITING,
        RP.EXIT_STOPPED_POD_COLLISION,
        EXIT_CPU_LANE_DRY,
    }


# ---------------------------------------------------------------------------
# Executed shell sandbox (bootstrap_pod.sh step 10 → EOF) — Test 8
# ---------------------------------------------------------------------------


def _exec_step10_region(text: str, ssh_rc: int) -> tuple[subprocess.CompletedProcess, int]:
    """Run the REAL step-10-through-EOF region in a sandboxed bash.

    Prelude stubs only the boundaries: ``ssh_cmd`` (returns ``ssh_rc``) and
    the step/log helpers; the constant line is extracted VERBATIM from the
    script so the sandbox binds the real value. Runs under the production
    ``set -euo pipefail`` regime, so the region's own ``set +e``/``set -e``
    rc-capture bracketing is exercised for real. Returns (proc, constant).
    """
    region = text[text.index("# ── Step 10: Preflight check") :]
    m = _CONST_RE.search(text)
    assert m is not None, "EXIT_PREFLIGHT_FAILED assignment not found"
    prelude = "\n".join(
        [
            "set -euo pipefail",
            "NO_PREFLIGHT=false",
            "POD_NAME=sandboxpod",
            "HOST=h",
            "PORT=1",
            "REMOTE_DIR=/workspace/explore-persona-space",
            "GREEN=''; BOLD=''; NC=''",
            m.group(0),  # the REAL constant assignment, verbatim
            "step() { :; }",
            'log_ok() { echo "OK: $*"; }',
            "log_warn() { :; }",
            'log_fail() { echo "FAIL: $*"; }',
            f"ssh_cmd() {{ return {ssh_rc}; }}",
            "",
        ]
    )
    proc = subprocess.run(
        ["bash", "-c", prelude + region],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc, int(m.group(1))


def test_step10_sandbox_preflight_failure_exits_78_with_local_sentinel(tmp_path) -> None:
    """(Test 8, failure arm) a preflight rc=2 exits the script with the
    mapped 78, prints the sentinel LOCALLY with the RAW rc, and never
    reaches the ``Bootstrap complete`` summary."""
    proc, const = _exec_step10_region(_text(), ssh_rc=2)
    assert proc.returncode == const == 78, (proc.returncode, proc.stdout, proc.stderr)
    assert "PREFLIGHT-FAILED-AT-BOOTSTRAP rc=2" in proc.stdout, (proc.stdout, proc.stderr)
    assert "FAIL: Preflight FAILED (rc=2)" in proc.stdout, proc.stdout
    assert "Bootstrap FAILED at preflight for sandboxpod (preflight rc=2)" in proc.stdout
    assert "Bootstrap complete" not in proc.stdout, proc.stdout
    assert "OK: Preflight passed" not in proc.stdout, proc.stdout


def test_step10_sandbox_preflight_pass_exits_0_reaches_summary(tmp_path) -> None:
    """(Test 8, success arm) a preflight rc=0 completes the region: exit 0,
    no sentinel, the summary banner reached."""
    proc, _const = _exec_step10_region(_text(), ssh_rc=0)
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert "OK: Preflight passed" in proc.stdout, proc.stdout
    assert "PREFLIGHT-FAILED-AT-BOOTSTRAP" not in proc.stdout, proc.stdout
    assert "Bootstrap complete for sandboxpod" in proc.stdout, proc.stdout


# ---------------------------------------------------------------------------
# Launch-path rc-78 disposition (backends/runpod.py) — Test 9
# ---------------------------------------------------------------------------


def _spec(*, extra: dict | None = None, **overrides) -> RunSpec:
    return RunSpec(
        issue=2606,
        intent="lora-7b",
        backend="runpod",
        workload_cmd="bash scripts/issue2606_dispatch.sh",
        extra=extra or {},
        **overrides,
    )


def _relay_raising(rc: int, raised: list) -> object:
    """A relay stub raising PodLifecycleProcessError(rc), recording the
    instance so identity (ORIGINAL re-raised, never a copy) is assertable."""

    def _relay(cmd, **_k):
        err = RP.PodLifecycleProcessError(
            rc,
            cmd,
            output=None,
            stderr=f"BOOTSTRAP-FAILED pod=pod-2606 rc={rc} reason=preflight\n",
        )
        raised.append(err)
        raise err

    return _relay


def test_launch_rc78_terminates_by_exact_id_and_reraises_original(monkeypatch):
    """(Test 9) an rc=78 provision relay failure best-effort terminates the
    just-provisioned pod by EXACT id (#2038-family disposition) and
    re-raises the ORIGINAL PodLifecycleProcessError instance."""
    raised: list = []
    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", _relay_raising(78, raised))
    looked_up: list[str] = []
    monkeypatch.setattr(
        RP,
        "_provisioned_pod_id",
        lambda pod_name: looked_up.append(pod_name) or "rpPRE78",
    )
    calls: list[dict] = []
    monkeypatch.setattr(
        RP,
        "_terminate_just_provisioned",
        lambda **kw: calls.append(kw) or True,
    )
    with pytest.raises(RP.PodLifecycleProcessError) as ei:
        RP.RunPodBackend().launch(_spec())
    assert ei.value is raised[0]  # the ORIGINAL error, never a re-wrap
    assert ei.value.returncode == 78
    assert looked_up == ["pod-2606"]  # id resolved for the RECOMPUTED pod name
    assert len(calls) == 1
    assert calls[0]["pod_id"] == "rpPRE78"
    assert calls[0]["pod_name"] == "pod-2606"
    assert calls[0]["issue"] == 2606
    assert "preflight" in calls[0]["cause"]
    assert "#2606" in calls[0]["cause"]


def test_launch_rc78_suffixed_lane_recomputes_suffixed_pod_name(monkeypatch):
    """(Test 9, suffix arm) a lane_suffix-bearing spec terminates the
    SUFFIXED pod name — pod-2606-b, never the bare pod-2606."""
    raised: list = []
    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", _relay_raising(78, raised))
    monkeypatch.setattr(RP, "_provisioned_pod_id", lambda pod_name: "rpPRE78b")
    calls: list[dict] = []
    monkeypatch.setattr(
        RP,
        "_terminate_just_provisioned",
        lambda **kw: calls.append(kw) or True,
    )
    with pytest.raises(RP.PodLifecycleProcessError):
        RP.RunPodBackend().launch(_spec(extra={"lane_suffix": "b"}))
    assert len(calls) == 1
    assert calls[0]["pod_name"] == "pod-2606-b"


def test_launch_rc78_teardown_raise_never_masks_original(monkeypatch, caplog):
    """(Test 9, mask guard) even a RAISING teardown never masks the ORIGINAL
    rc=78 relay error (the _dispose_post_provision_failure mask-guard
    shape)."""
    raised: list = []
    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", _relay_raising(78, raised))
    monkeypatch.setattr(RP, "_provisioned_pod_id", lambda pod_name: "rpPRE78")

    def _teardown_boom(**_kw):
        raise RuntimeError("teardown exploded")

    monkeypatch.setattr(RP, "_terminate_just_provisioned", _teardown_boom)
    with (
        caplog.at_level(logging.ERROR, logger="explore_persona_space.backends.runpod"),
        pytest.raises(RP.PodLifecycleProcessError) as ei,
    ):
        RP.RunPodBackend().launch(_spec())
    assert ei.value is raised[0]
    assert "emergency teardown wrapper itself raised" in caplog.text
    assert "rc=78" in caplog.text


def test_launch_non78_relay_failure_propagates_untouched(monkeypatch):
    """(Test 9, control arm) a non-78 relay failure (rc=100) takes NO
    teardown action — the error propagates exactly as pre-#2606 (the
    exit-75 still-waiting contract rides the same bare re-raise)."""
    raised: list = []
    monkeypatch.setattr(RP, "_run_pod_lifecycle_relay", _relay_raising(100, raised))
    monkeypatch.setattr(RP, "_provisioned_pod_id", _explode)
    monkeypatch.setattr(RP, "_terminate_just_provisioned", _explode)
    with pytest.raises(RP.PodLifecycleProcessError) as ei:
        RP.RunPodBackend().launch(_spec())
    assert ei.value is raised[0]
    assert ei.value.returncode == 100
