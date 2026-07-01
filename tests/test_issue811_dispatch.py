"""Issue #811 dispatcher-level contract tests.

Bind the PRE-SPEND HALT contract at the DISPATCHER level, not just the fit-code
level: a phase0-gate exit 3 (KILL-1 FIRE) MUST HALT the dispatcher BEFORE any
``issue667_extract.py --turn-nl`` full paired re-extraction runs (the ~7 GPU-h
spend). Round-2 fixed the fit-code decision; round-3 adds this dispatcher binding
so a future refactor that reorders the phases can never let the FIRE be ignored.

Also pins the round-3 BLOCKER fix (phase0-gate-reads-unuploaded-hf-store): a
production run passes ``--local-root eval_results/issue_811/phase0_base_leg`` to the
phase0-gate (the store is on disk at gate time, NOT yet on HF).

Mechanics: the tests run the REAL ``scripts/issue811_dispatch.sh`` inside a sandbox
``$WORKLOAD_ROOT`` whose ``scripts/`` holds trivial stubs; a fake ``uv`` early on
``PATH`` logs every ``uv run python scripts/<name> ...`` invocation to a file and
returns a per-script exit code the test controls. No GPU, no network, no real
``uv`` — the shell control-flow (phase ordering, rc capture, HALT) is what is under
test.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DISPATCH = PROJECT_ROOT / "scripts" / "issue811_dispatch.sh"


def _make_fake_uv(sandbox: Path, invocation_log: Path, gate_rc: int) -> Path:
    """Write a fake `uv` on PATH: logs every `uv run python scripts/<name> ...` and
    returns `gate_rc` for the phase0-gate invocation (0 for everything else)."""
    bindir = sandbox / "fakebin"
    bindir.mkdir(parents=True, exist_ok=True)
    uv = bindir / "uv"
    # The dispatcher calls `uv run python scripts/<name> ...` and `uv run python - <<PY`.
    # Log the full arg vector; special-case the phase0-gate (issue811_fit.py --phase0-gate)
    # to return gate_rc so the test drives the KILL-1 FIRE / PASS branch.
    uv.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> {invocation_log}\n'
        "# `uv run python - ` (heredoc sentinel writer) — consume stdin, succeed.\n"
        'if [ "$1" = "run" ] && [ "$3" = "python" ] && [ "$4" = "-" ]; then\n'
        "  cat > /dev/null; exit 0\n"
        "fi\n"
        "# phase0-gate: issue811_fit.py --phase0-gate -> return the controlled rc.\n"
        'case "$*" in\n'
        f"  *issue811_fit.py*--phase0-gate*) exit {gate_rc} ;;\n"
        "esac\n"
        "exit 0\n"
    )
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC | stat.S_IRWXU)
    return bindir


def _run_dispatch(
    tmp_path: Path, gate_rc: int, extra_args: list[str], *, precreate_phase0: bool = True
) -> tuple[int, str]:
    """Run the real dispatcher in a sandbox with a fake `uv`. Returns (rc, invocation_log).

    ``precreate_phase0`` seeds a phase0 store on disk so the dispatcher's
    ``-d "$PHASE0_DIR"`` local-root branch fires (the round-3 fix). Set it False to
    reproduce the round-4 case: a fresh NON-skip run whose Phase 0 produced nothing
    (the fake `uv` extractor creates no dir), which MUST hard-fail (exit 5) rather
    than fall back to HF."""
    sandbox = tmp_path / "workload"
    (sandbox / "scripts").mkdir(parents=True, exist_ok=True)
    if precreate_phase0:
        # One dummy file is enough for `-d` to be true.
        phase0 = sandbox / "eval_results" / "issue_811" / "phase0_base_leg"
        phase0.mkdir(parents=True, exist_ok=True)
        (phase0 / "marker.txt").write_text("x")
    invocation_log = tmp_path / "invocations.log"
    invocation_log.write_text("")
    bindir = _make_fake_uv(sandbox, invocation_log, gate_rc)
    env = dict(os.environ)
    env["WORKLOAD_ROOT"] = str(sandbox)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env["EPM_SKIP_UPLOAD"] = "1"  # never attempt a real upload
    proc = subprocess.run(
        ["bash", str(DISPATCH), "--sources", "default,sp_swe", *extra_args],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.returncode, invocation_log.read_text()


def test_dispatcher_halts_before_phase1_on_kill1_fire(tmp_path):
    """phase0-gate exit 3 (KILL-1 FIRE) -> dispatcher exits 3 and NEVER runs Phase 1."""
    rc, log = _run_dispatch(tmp_path, gate_rc=3, extra_args=[])
    assert rc == 3, f"dispatcher rc={rc}, expected 3 (KILL-1 HALT)\nlog:\n{log}"
    # The gate WAS reached...
    assert "issue811_fit.py --phase0-gate" in log
    # ...and the ~7 GPU-h Phase-1 paired re-extraction was NEVER invoked.
    assert "issue667_extract.py" not in log, f"Phase 1 ran after KILL-1 FIRE!\nlog:\n{log}"
    assert "--turn-nl" not in log, f"paired --turn-nl extract ran after KILL-1 FIRE!\nlog:\n{log}"


def test_dispatcher_halts_before_phase1_on_nonzero_gate(tmp_path):
    """A non-3 gate failure (e.g. the fail-loud degenerate-pass RuntimeError -> rc=1)
    ALSO halts before Phase 1 — the dispatcher propagates any non-zero gate rc."""
    rc, log = _run_dispatch(tmp_path, gate_rc=1, extra_args=[])
    assert rc == 1, f"dispatcher rc={rc}, expected 1 (non-KILL-1 gate failure HALT)\nlog:\n{log}"
    assert "issue811_fit.py --phase0-gate" in log
    assert "issue667_extract.py" not in log, f"Phase 1 ran after a gate failure!\nlog:\n{log}"


def test_dispatcher_reaches_phase1_on_gate_pass(tmp_path):
    """phase0-gate exit 0 (KILL-1 PASS) -> Phase 1 paired --turn-nl extract DOES run."""
    rc, log = _run_dispatch(tmp_path, gate_rc=0, extra_args=[])
    assert rc == 0, f"dispatcher rc={rc}, expected 0 (clean PASS-through)\nlog:\n{log}"
    assert "issue811_fit.py --phase0-gate" in log
    # Phase 1 IS reached on a PASS (the contrapositive — proves the HALT tests aren't
    # vacuously passing because Phase 1 never runs).
    assert "issue667_extract.py" in log and "--turn-nl" in log, (
        f"Phase 1 paired extract did NOT run on a gate PASS!\nlog:\n{log}"
    )


def test_dispatcher_passes_local_root_phase0_dir_to_gate(tmp_path):
    """The phase0-gate is invoked with --local-root <PHASE0_DIR> in the production path
    (round-3 BLOCKER phase0-gate-reads-unuploaded-hf-store): the store is on disk at
    gate time, NOT yet on HF, so the gate MUST read the local mirror."""
    _rc, log = _run_dispatch(tmp_path, gate_rc=0, extra_args=[])
    gate_lines = [ln for ln in log.splitlines() if "issue811_fit.py --phase0-gate" in ln]
    assert gate_lines, f"no phase0-gate invocation logged\nlog:\n{log}"
    assert "--local-root eval_results/issue_811/phase0_base_leg" in gate_lines[0], (
        f"phase0-gate not pointed at the local phase0 store\ngate line:\n{gate_lines[0]}"
    )


def test_dispatcher_hard_fails_when_local_store_missing_on_nonskip_run(tmp_path):
    """FRESH NON-skip run whose Phase 0 produced NO local store -> HARD-FAIL (exit 5)
    BEFORE the phase0-gate runs; NEVER falls back to HF (round-4 BLOCKER
    phase0-hf-fallback-not-skip-gated).

    The fake `uv` extractor creates no dir, so $PHASE0_DIR is absent at gate-selection
    time on a non-skip run. The dispatcher must refuse to read the HF prefix (a
    stale/other-run/empty store) and halt with exit 5 — and the phase0-gate + Phase 1
    must NEVER run."""
    rc, log = _run_dispatch(tmp_path, gate_rc=0, extra_args=[], precreate_phase0=False)
    assert rc == 5, f"dispatcher rc={rc}, expected 5 (missing-local-store HALT)\nlog:\n{log}"
    # The gate itself was NEVER invoked (the HF-fallback guard fired first).
    assert "issue811_fit.py --phase0-gate" not in log, (
        f"phase0-gate ran despite the missing local store!\nlog:\n{log}"
    )
    # ...and Phase 1 paired extract certainly never ran.
    assert "issue667_extract.py" not in log, f"Phase 1 ran after a missing-store HALT!\nlog:\n{log}"


def test_dispatcher_skip_extract_allows_hf_fallback_when_no_local_store(tmp_path):
    """--skip-extract resume with NO local store IS allowed to fall back to the HF
    prefix (empty --local-root) — the store is on HF from the prior run, and the
    fit-side empty-store guard catches a vacuous HF tree. The gate MUST still run
    (contrapositive: the exit-5 guard is scoped to NON-skip runs only)."""
    rc, log = _run_dispatch(
        tmp_path, gate_rc=0, extra_args=["--skip-extract"], precreate_phase0=False
    )
    assert rc == 0, f"dispatcher rc={rc}, expected 0 (--skip-extract HF-fallback PASS)\nlog:\n{log}"
    gate_lines = [ln for ln in log.splitlines() if "issue811_fit.py --phase0-gate" in ln]
    assert gate_lines, f"phase0-gate did NOT run on a --skip-extract resume\nlog:\n{log}"
    # No --local-root on the HF-fallback path (empty PHASE0_LOCAL_ARGS).
    assert "--local-root" not in gate_lines[0], (
        f"--skip-extract HF fallback wrongly passed --local-root\ngate line:\n{gate_lines[0]}"
    )
