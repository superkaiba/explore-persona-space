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


def _make_fake_uv(
    sandbox: Path, invocation_log: Path, gate_rc: int, *, stage_creates_store: bool = False
) -> Path:
    """Write a fake `uv` on PATH: logs every `uv run python scripts/<name> ...` and
    returns `gate_rc` for the phase0-gate invocation (0 for everything else).

    ``stage_creates_store``: when True, the fake stage script (issue811_stage_phase0.py)
    parses its ``--out`` arg and CREATES that dir (with a marker file), simulating a
    real successful stage that makes $PHASE0_DIR available on disk. This lets the
    stage-skip test run with precreate_phase0=False and still exercise the gate's
    ``-d "$PHASE0_DIR"`` local-root branch — proving the STAGE STEP ITSELF (not a
    test-seeded dir) is what makes the store available downstream."""
    bindir = sandbox / "fakebin"
    bindir.mkdir(parents=True, exist_ok=True)
    uv = bindir / "uv"
    # The dispatcher calls `uv run python scripts/<name> ...` and `uv run python - <<PY`.
    # Log the full arg vector; special-case the phase0-gate (issue811_fit.py --phase0-gate)
    # to return gate_rc so the test drives the KILL-1 FIRE / PASS branch.
    stage_block = (
        "# stage script: emulate a successful stage that materializes $PHASE0_DIR.\n"
        'if [[ "$*" == *issue811_stage_phase0.py* ]]; then\n'
        '  out=""; prev=""\n'
        '  for a in "$@"; do [ "$prev" = "--out" ] && out="$a"; prev="$a"; done\n'
        '  [ -n "$out" ] && mkdir -p "$out" && echo staged > "$out/marker.txt"\n'
        "  exit 0\n"
        "fi\n"
        if stage_creates_store
        else ""
    )
    uv.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> {invocation_log}\n'
        "# `uv run python - ` (heredoc sentinel writer) — consume stdin, succeed.\n"
        'if [ "$1" = "run" ] && [ "$3" = "python" ] && [ "$4" = "-" ]; then\n'
        "  cat > /dev/null; exit 0\n"
        "fi\n"
        f"{stage_block}"
        "# phase0-gate: issue811_fit.py --phase0-gate -> return the controlled rc.\n"
        'case "$*" in\n'
        f"  *issue811_fit.py*--phase0-gate*) exit {gate_rc} ;;\n"
        "esac\n"
        "exit 0\n"
    )
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC | stat.S_IRWXU)
    return bindir


def _run_dispatch(
    tmp_path: Path,
    gate_rc: int,
    extra_args: list[str],
    *,
    precreate_phase0: bool = True,
    stage_prefix: str | None = None,
    stage_creates_store: bool = False,
    variant: str | None = None,
) -> tuple[int, str]:
    """Run the real dispatcher in a sandbox with a fake `uv`. Returns (rc, invocation_log).

    ``precreate_phase0`` seeds a phase0 store on disk so the dispatcher's
    ``-d "$PHASE0_DIR"`` local-root branch fires (the round-3 fix). Set it False to
    reproduce the round-4 case: a fresh NON-skip run whose Phase 0 produced nothing
    (the fake `uv` extractor creates no dir), which MUST hard-fail (exit 5) rather
    than fall back to HF. ``stage_prefix`` sets EPM_PHASE0_STAGE_PREFIX to exercise
    the round-6 phase-0 staging branch (the fake `uv` returns 0 for the stage script,
    simulating a successful + complete stage — the real stage script would fail loud
    on a shortfall, covered by the unit tests in test_issue811_turn_nl.py)."""
    sandbox = tmp_path / "workload"
    (sandbox / "scripts").mkdir(parents=True, exist_ok=True)
    if precreate_phase0:
        # One dummy file is enough for `-d` to be true. The maxp arm uses the
        # round's OWN dirs (SUMMARY_VARIANT case in the dispatcher).
        round_dir = ("issue_811", "maxp-winner-mapchange") if variant == "maxp" else ("issue_811",)
        phase0 = sandbox.joinpath("eval_results", *round_dir, "phase0_base_leg")
        phase0.mkdir(parents=True, exist_ok=True)
        (phase0 / "marker.txt").write_text("x")
    invocation_log = tmp_path / "invocations.log"
    invocation_log.write_text("")
    bindir = _make_fake_uv(
        sandbox, invocation_log, gate_rc, stage_creates_store=stage_creates_store
    )
    env = dict(os.environ)
    env["WORKLOAD_ROOT"] = str(sandbox)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env["EPM_SKIP_UPLOAD"] = "1"  # never attempt a real upload
    if stage_prefix is not None:
        env["EPM_PHASE0_STAGE_PREFIX"] = stage_prefix
    else:
        env.pop("EPM_PHASE0_STAGE_PREFIX", None)
    if variant is not None:
        env["SUMMARY_VARIANT"] = variant
    else:
        env.pop("SUMMARY_VARIANT", None)
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


def test_dispatcher_stages_phase0_and_skips_reextraction(tmp_path):
    """round-6/round-7: EPM_PHASE0_STAGE_PREFIX set -> the dispatcher stages the
    completed phase-0 store (invokes issue811_stage_phase0.py), SKIPS the ~5.6h
    base-leg re-extraction, and still runs the parity + gate + Phase 1 paired extract.

    precreate_phase0=False + stage_creates_store=True proves the STAGE STEP ITSELF is
    what makes $PHASE0_DIR available: the phase0 dir is NOT test-seeded, so the ONLY
    way the gate's ``-d "$PHASE0_DIR"`` local-root branch can fire is if the fake stage
    script materialized the store from its --out arg (round-7 Minor #2 — the prior
    version seeded the dir, so it could not distinguish "stage made it" from "test
    made it"). The fake `uv` phase-0 EXTRACTOR still creates nothing, so a passing gate
    proves the extraction was genuinely skipped in favor of the staged store."""
    rc, log = _run_dispatch(
        tmp_path,
        gate_rc=0,
        extra_args=[],
        precreate_phase0=False,  # NOT test-seeded — the stage step must materialize it
        stage_creates_store=True,  # fake stage script creates $PHASE0_DIR from --out
        stage_prefix="issue811_partial/att-20260701-233116/eval_results_issue_811/phase0_base_leg",
    )
    assert rc == 0, f"dispatcher rc={rc}, expected 0 (staged PASS-through)\nlog:\n{log}"
    # The stage script WAS invoked with the prefix...
    stage_lines = [ln for ln in log.splitlines() if "issue811_stage_phase0.py" in ln]
    assert stage_lines, f"stage script never invoked despite EPM_PHASE0_STAGE_PREFIX\nlog:\n{log}"
    assert "att-20260701-233116" in stage_lines[0]
    # ...the phase-0 base-leg re-extractor was NEVER run (the ~5.6h skip)...
    assert "issue811_phase0_extract.py" not in log, (
        f"phase-0 base-leg re-extraction ran despite a staged store!\nlog:\n{log}"
    )
    # ...the gate ran with --local-root (proving the stage step made $PHASE0_DIR
    # present — it was NOT test-seeded; a missing store would have hit exit 5)...
    gate_lines = [ln for ln in log.splitlines() if "issue811_fit.py --phase0-gate" in ln]
    assert gate_lines, f"gate skipped after staging!\nlog:\n{log}"
    assert "--local-root eval_results/issue_811/phase0_base_leg" in gate_lines[0], (
        f"gate not pointed at the STAGED local store — stage step didn't materialize "
        f"$PHASE0_DIR\ngate line:\n{gate_lines[0]}"
    )
    # ...and parity + Phase 1 paired extract all still run on the staged store.
    assert "issue811_mean_parity_check.py" in log, f"parity skipped after staging!\nlog:\n{log}"
    assert "issue667_extract.py" in log and "--turn-nl" in log, (
        f"Phase 1 paired extract did NOT run after a staged PASS!\nlog:\n{log}"
    )


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


def test_dispatcher_maxp_arm_wires_summary_flags(tmp_path):
    """SUMMARY_VARIANT=maxp env/flag wiring pin (r10 Minor): the maxp arm points the
    gate at --test-summary maxp + the round's OWN phase0 dir, adds --maxp to every
    Phase-1 extract, runs three-summary fits, and runs the maxp-only phases (3b
    committed-mean parity, 4b F1 offset decomposition)."""
    rc, log = _run_dispatch(tmp_path, gate_rc=0, extra_args=[], variant="maxp")
    assert rc == 0, f"dispatcher rc={rc}, expected 0 (maxp arm PASS-through)\nlog:\n{log}"
    gate_lines = [ln for ln in log.splitlines() if "issue811_fit.py --phase0-gate" in ln]
    assert gate_lines, f"no phase0-gate invocation logged\nlog:\n{log}"
    assert "--test-summary maxp" in gate_lines[0], (
        f"maxp arm gate not re-pointed at --test-summary maxp\ngate line:\n{gate_lines[0]}"
    )
    assert (
        "--local-root eval_results/issue_811/maxp-winner-mapchange/phase0_base_leg" in gate_lines[0]
    ), f"maxp gate not reading the round's OWN phase0 store\ngate line:\n{gate_lines[0]}"
    ext_lines = [ln for ln in log.splitlines() if "issue667_extract.py" in ln]
    assert ext_lines and all("--turn-nl --maxp" in ln for ln in ext_lines), (
        f"Phase-1 extract missing '--turn-nl --maxp'\nextract lines:\n{ext_lines}\nlog:\n{log}"
    )
    fit_lines = [
        ln for ln in log.splitlines() if "issue811_fit.py" in ln and "--phase0-gate" not in ln
    ]
    assert fit_lines and "--summaries mean turn_nl maxp" in fit_lines[0], (
        f"Phase-3 fits not three-summary\nfit lines:\n{fit_lines}"
    )
    assert "--compare-committed" in log, f"maxp-only Phase 3b parity read never ran\nlog:\n{log}"
    assert "issue811_offset_decomposition.py" in log, (
        f"maxp-only Phase 4b F1 offset decomposition never ran\nlog:\n{log}"
    )


def test_dispatcher_maxp_arm_refuses_stage_prefix(tmp_path):
    """SUMMARY_VARIANT=maxp + EPM_PHASE0_STAGE_PREFIX -> exit 2 BEFORE any phase runs
    (no prior phase-0 store carries v0_maxp; plan §4 item 5 fail-fast guard)."""
    rc, log = _run_dispatch(
        tmp_path,
        gate_rc=0,
        extra_args=[],
        variant="maxp",
        stage_prefix="issue811_partial/att-20260701-233116/eval_results_issue_811/phase0_base_leg",
    )
    assert rc == 2, f"dispatcher rc={rc}, expected 2 (maxp stage-prefix refusal)\nlog:\n{log}"
    assert log.strip() == "", f"phases ran despite the maxp stage-prefix refusal\nlog:\n{log}"
