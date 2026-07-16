"""Dispatch-time lane-env lint for ``--workload-cmd`` (#1329, incident #825).

Pins the pure lint (`backends/issue_dispatch.lint_workload_cmd_lane_env` +
`LANE_WORKLOAD_ENV_EXPORTS`), the `dispatch_issue.py launch` wiring
(warn-by-default + `extra.workload_cmd_lane_env_risk` marker flag, exit-2
pre-route refusal on a provably-certain lane or under
`--strict-workload-cmd-env`, the `EPM_SKIP_WORKLOAD_CMD_ENV_LINT=1` kill
switch), the `backend_poll._runspec_from_gcp_handle` failover-time warn-only
breadcrumb, the forward/inverse parity of the curated per-lane export map
against the actual renderer sources, and the SKILL.md Step 6b rule (f)
durability pin.

Incident #825 (Track-S): a GCP crash failed over to RunPod; the reused
workload-cmd ``REPO_ROOT="$WORKLOAD_ROOT" bash scripts/...`` aborted under
the RunPod launcher's ``set -uo pipefail`` because ``WORKLOAD_ROOT`` is
exported only by the GCP startup script.
"""

from __future__ import annotations

import io
import json
import logging
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from explore_persona_space.backends.base import RunHandle
from explore_persona_space.backends.issue_dispatch import (
    LANE_WORKLOAD_ENV_EXPORTS,
    lint_workload_cmd_lane_env,
)
from tests.test_dispatch_issue_cli import (
    _backend_selected_extras,
    _build_mock_factory,
    _cd_to_tmp,
    _MockBackend,
)

REPO = Path(__file__).resolve().parent.parent

#: The verbatim #825 incident command (preserved in commit 65ff2426a8's diff).
INCIDENT_825_CMD = 'REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue825_naturalistic_s_dispatch.sh'
#: The post-fix #825 command (driver self-resolves REPO_ROOT).
FIXED_825_CMD = "bash scripts/issue825_naturalistic_s_dispatch.sh"
#: The set-u-safe lane-portable inline form the lint recommends.
DEFAULTED_CMD = 'REPO_ROOT="${WORKLOAD_ROOT:-$PWD}" bash scripts/x.sh'


# ---------------------------------------------------------------------------
# Pure lint (plan #1329 §4 tests 1-10 + critic-round additions)
# ---------------------------------------------------------------------------


def test_incident_825_cmd_flags_workload_root_on_auto() -> None:
    """Plan test 1: the verbatim #825 trace flags WORKLOAD_ROOT, warn-class."""
    r = lint_workload_cmd_lane_env(INCIDENT_825_CMD, backend_value="auto")
    assert r.flagged == {"WORKLOAD_ROOT": ("runpod", "slurm")}
    assert r.certain == ()
    assert r.reachable_lanes == ("gcp", "runpod", "slurm")


def test_fixed_825_cmd_passes_clean() -> None:
    r = lint_workload_cmd_lane_env(FIXED_825_CMD, backend_value="auto")
    assert not r.flagged
    assert not r.certain


def test_brace_form_flags_and_longer_name_does_not() -> None:
    """Plan test 2: ``${VAR}`` bare-brace flags; ``$VARX`` does not."""
    r = lint_workload_cmd_lane_env("echo ${WORKLOAD_ROOT}", backend_value="auto")
    assert "WORKLOAD_ROOT" in r.flagged
    r2 = lint_workload_cmd_lane_env("echo $WORKLOAD_ROOTX", backend_value="auto")
    assert not r2.flagged


@pytest.mark.parametrize(
    "form",
    [
        "${WORKLOAD_ROOT:-w}",
        "${WORKLOAD_ROOT-w}",
        "${WORKLOAD_ROOT:+w}",
        "${WORKLOAD_ROOT+w}",
        "${WORKLOAD_ROOT:=w}",
    ],
)
def test_set_u_safe_expansions_not_flagged(form: str) -> None:
    """Plan test 3 (+ critic parametrization): defaulted/alternate/assign
    expansions are set-u-safe by POSIX and are never flagged."""
    r = lint_workload_cmd_lane_env(f'REPO_ROOT="{form}" bash s.sh', backend_value="auto")
    assert not r.flagged, (form, r)


def test_assignment_and_escaped_not_flagged() -> None:
    """Plan test 4: an assignment has no ``$``; ``\\$VAR`` is escaped."""
    r = lint_workload_cmd_lane_env("WORKLOAD_ROOT=/x bash s.sh", backend_value="auto")
    assert not r.flagged
    r2 = lint_workload_cmd_lane_env(r"echo \$WORKLOAD_ROOT", backend_value="auto")
    assert not r2.flagged


def test_single_quoted_literal_not_flagged() -> None:
    """Plan test 5: POSIX single quotes contain no expansion."""
    r = lint_workload_cmd_lane_env("echo '$WORKLOAD_ROOT'", backend_value="auto")
    assert not r.flagged


def test_quote_strip_boundary_apostrophe_inside_double_quotes() -> None:
    """Critic addition (2): pin the DOCUMENTED quote-strip boundary behavior.

    An unpaired apostrophe inside double quotes leaves the single-quote
    stripper with no match, so the bare ``$WORKLOAD_ROOT`` stays visible and
    FLAGS (the conservative direction). The conservative-direction claim is
    NOT universal: an apostrophe can pair with a LATER genuine single-quote
    opener, stripping the region between them and hiding a genuinely-bare
    reference (a documented false NEGATIVE, pinned below).
    """
    r = lint_workload_cmd_lane_env('bash -c "don\'t touch $WORKLOAD_ROOT"', backend_value="auto")
    assert "WORKLOAD_ROOT" in r.flagged
    # The documented false-negative direction: the apostrophe in "don't"
    # pairs with the opener of the trailing 'x' literal and the bare
    # reference between them is stripped.
    r2 = lint_workload_cmd_lane_env("echo \"don't\" $WORKLOAD_ROOT 'x'", backend_value="auto")
    assert not r2.flagged


def test_nondefaulting_expansions_are_deliberate_v1_false_negatives() -> None:
    """Critic addition (3): ``${V%pat}`` / ``${V#pat}`` / ``${V:0:3}`` DO
    abort under ``set -u`` on an unbound var but are deliberately NOT matched
    in v1 (rare in workload-cmd strings; widen if one ever bites). Pinned so
    a future widening is a conscious contract change."""
    for form in ("${WORKLOAD_ROOT%pat}", "${WORKLOAD_ROOT#pat}", "${WORKLOAD_ROOT:0:3}"):
        r = lint_workload_cmd_lane_env(f"echo {form}", backend_value="auto")
        assert not r.flagged, form


def test_subshell_and_backtick_references_flag() -> None:
    """Critic addition (6): a bare reference inside ``$(...)`` or backticks
    is still a bare reference (command substitution runs under the same
    ``set -u`` shell)."""
    r = lint_workload_cmd_lane_env(
        "RESULT=$(compute $WORKLOAD_ROOT) bash s.sh", backend_value="auto"
    )
    assert "WORKLOAD_ROOT" in r.flagged
    r2 = lint_workload_cmd_lane_env("echo `ls $WORKLOAD_ROOT`", backend_value="auto")
    assert "WORKLOAD_ROOT" in r2.flagged


def test_runpod_execute_workload_is_certain_provision_only_is_not() -> None:
    """Plan test 6: explicit runpod + --execute-workload provably executes
    the cmd under the launcher's set -uo pipefail (runpod.py:504) → certain;
    a provision-only runpod launch downgrades to warn."""
    r = lint_workload_cmd_lane_env(
        "echo $WORKLOAD_ROOT", backend_value="runpod", execute_workload=True
    )
    assert r.certain == ("WORKLOAD_ROOT",)
    assert r.reachable_lanes == ("runpod",)
    r2 = lint_workload_cmd_lane_env(
        "echo $WORKLOAD_ROOT", backend_value="runpod", execute_workload=False
    )
    assert r2.flagged == {"WORKLOAD_ROOT": ("runpod",)}
    assert r2.certain == ()


def test_gcp_backend_flags_runpod_failover_risk_not_certain() -> None:
    """Plan test 7: explicit gcp reaches {gcp, runpod} (the Part B workload
    failover reuses the cmd on RunPod) — flagged missing (runpod,), never
    certain. ``$SCRATCH_JOB_DIR`` left the candidate universe with the
    fact-check correction (unexported on EVERY lane → behaves like any
    arbitrary ``$FOO``; covered by the never-flagged test below)."""
    r = lint_workload_cmd_lane_env("echo $WORKLOAD_ROOT", backend_value="gcp")
    assert r.flagged == {"WORKLOAD_ROOT": ("runpod",)}
    assert r.certain == ()
    assert r.reachable_lanes == ("gcp", "runpod")


@pytest.mark.parametrize("backend", ["nibi", "fir", "mila", "cluster"])
def test_explicit_slurm_lane_is_certain(backend: str) -> None:
    """Plan test 8: the SLURM custom stage executes the cmd via the literal
    ``bash -eu -o pipefail -c`` append (slurm.py:1577) → certain. The legacy
    ``cluster`` alias normalizes to nibi."""
    r = lint_workload_cmd_lane_env("echo $WORKLOAD_ROOT", backend_value=backend)
    assert r.certain == ("WORKLOAD_ROOT",)
    assert r.reachable_lanes == ("slurm",)


def test_eps_vars_missing_lane_sets_on_auto() -> None:
    """Plan test 9: EPS_SENTINEL_PATH is gcp-only; EPS_ISSUE is gcp+slurm."""
    r = lint_workload_cmd_lane_env("echo $EPS_SENTINEL_PATH", backend_value="auto")
    assert r.flagged == {"EPS_SENTINEL_PATH": ("runpod", "slurm")}
    r2 = lint_workload_cmd_lane_env("echo $EPS_ISSUE", backend_value="auto")
    assert r2.flagged == {"EPS_ISSUE": ("runpod",)}


def test_universal_and_out_of_universe_vars_never_flagged() -> None:
    """Plan test 10: all-lane vars (WANDB_PROJECT/HOME/PATH) never flag;
    ``$FOO`` outside the universe (incl. the unexported SCRATCH_JOB_DIR)
    never flags; empty cmd → empty result."""
    for var in ("WANDB_PROJECT", "HOME", "PATH", "SOME_RANDOM_VAR", "SCRATCH_JOB_DIR"):
        r = lint_workload_cmd_lane_env(f"echo ${var}", backend_value="auto")
        assert not r.flagged, var
    empty = lint_workload_cmd_lane_env("", backend_value="auto")
    assert empty.flagged == {} and empty.certain == () and empty.reachable_lanes == ()


# ---------------------------------------------------------------------------
# CLI wiring (plan §4 tests 11-16 + critic additions 7/10)
# ---------------------------------------------------------------------------


def _guard_exploding_factory():
    raise AssertionError("backends_factory must not be called on a pre-route refusal")


def _lane_env_warnings(caplog) -> list[str]:
    return [
        rec.getMessage()
        for rec in caplog.records
        if rec.levelno >= logging.WARNING and "workload-cmd references" in rec.getMessage()
    ]


def test_launch_auto_flagged_cmd_warns_flags_marker_and_proceeds(
    monkeypatch, tmp_path, caplog
) -> None:
    """Plan test 11: auto + the #825 cmd → exit 0, backend launched, loud
    warning naming the var + BOTH lane-portable alternatives (critic 8), and
    ``extra.workload_cmd_lane_env_risk`` on the posted marker."""
    _cd_to_tmp(monkeypatch, tmp_path)
    nibi = _MockBackend(kind="nibi")
    marker_posts: list[dict] = []
    factory = _build_mock_factory(
        runpod=_MockBackend(kind="runpod"), nibi=nibi, marker_posts=marker_posts
    )

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            ["launch", "--issue", "825", "--intent", "lora-7b", "--workload-cmd", INCIDENT_825_CMD],
            backends_factory=factory,
        )
    assert rc == 0
    assert len(nibi.launches) == 1
    warnings = _lane_env_warnings(caplog)
    assert warnings, "expected a lane-env lint warning"
    joined = "\n".join(warnings)
    assert "WORKLOAD_ROOT" in joined
    # Critic 8: BOTH lane-portable alternatives appear (stable substrings).
    assert "${WORKLOAD_ROOT:-" in joined
    assert "${REPO_ROOT:-${WORKLOAD_ROOT:-" in joined
    extras = _backend_selected_extras(marker_posts)
    assert extras, "expected an epm:backend-selected post"
    assert all(
        e.get("workload_cmd_lane_env_risk") == {"WORKLOAD_ROOT": ["runpod", "slurm"]}
        for e in extras
    )


def test_launch_runpod_execute_workload_flagged_refuses_exit2_pre_route(
    monkeypatch, tmp_path
) -> None:
    """Plan test 12: explicit runpod + --execute-workload + flagged cmd →
    exit 2 BEFORE backends_factory (no pod provisioned)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "825",
                "--intent",
                "lora-7b",
                "--backend",
                "runpod",
                "--execute-workload",
                "--workload-cmd",
                INCIDENT_825_CMD,
            ],
            backends_factory=_guard_exploding_factory,
        )
    assert rc == 2
    body = json.loads(buf.getvalue().strip())
    assert body["ok"] is False
    assert body["reason"] == "workload_cmd_lane_env_unbound"
    assert body["failure_class"] == "infra"
    assert body["status"] == "blocked"
    assert "crash is certain" in body["note"]


def test_launch_explicit_slurm_flagged_refuses_exit2(monkeypatch, tmp_path) -> None:
    """Critic addition (10): an explicit SLURM lane executes the cmd under
    ``bash -eu -o pipefail -c`` → certain → exit-2 pre-route refusal."""
    _cd_to_tmp(monkeypatch, tmp_path)
    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "825",
                "--intent",
                "lora-7b",
                "--backend",
                "nibi",
                "--workload-cmd",
                INCIDENT_825_CMD,
            ],
            backends_factory=_guard_exploding_factory,
        )
    assert rc == 2
    body = json.loads(buf.getvalue().strip())
    assert body["reason"] == "workload_cmd_lane_env_unbound"


def test_launch_auto_defaulted_form_no_warning_no_flag(monkeypatch, tmp_path, caplog) -> None:
    """Plan test 13: the ``${WORKLOAD_ROOT:-$PWD}`` form passes untouched —
    existing GCP-lane dispatches stay unbroken."""
    _cd_to_tmp(monkeypatch, tmp_path)
    nibi = _MockBackend(kind="nibi")
    marker_posts: list[dict] = []
    factory = _build_mock_factory(
        runpod=_MockBackend(kind="runpod"), nibi=nibi, marker_posts=marker_posts
    )

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            ["launch", "--issue", "825", "--intent", "lora-7b", "--workload-cmd", DEFAULTED_CMD],
            backends_factory=factory,
        )
    assert rc == 0
    assert len(nibi.launches) == 1
    assert not _lane_env_warnings(caplog)
    for extra in _backend_selected_extras(marker_posts):
        assert "workload_cmd_lane_env_risk" not in extra


def test_launch_auto_flagged_strict_refuses_exit2(monkeypatch, tmp_path) -> None:
    """Plan test 14: --strict-workload-cmd-env upgrades a warn-class hit to
    the exit-2 pre-route refusal."""
    _cd_to_tmp(monkeypatch, tmp_path)
    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "825",
                "--intent",
                "lora-7b",
                "--strict-workload-cmd-env",
                "--workload-cmd",
                INCIDENT_825_CMD,
            ],
            backends_factory=_guard_exploding_factory,
        )
    assert rc == 2
    body = json.loads(buf.getvalue().strip())
    assert body["reason"] == "workload_cmd_lane_env_unbound"
    assert "--strict-workload-cmd-env" in body["note"]


def test_launch_strict_with_clean_cmd_proceeds(monkeypatch, tmp_path, caplog) -> None:
    """Critic addition (7): --strict-workload-cmd-env + a CLEAN cmd → exit 0,
    launch proceeds (the strict flag only bites on a flagged cmd)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(runpod=_MockBackend(kind="runpod"), nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "825",
                "--intent",
                "lora-7b",
                "--strict-workload-cmd-env",
                "--workload-cmd",
                FIXED_825_CMD,
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert len(nibi.launches) == 1
    assert not _lane_env_warnings(caplog)


def test_launch_hydra_lint_noop(monkeypatch, tmp_path, caplog) -> None:
    """Plan test 15: a --hydra launch has no workload_cmd → lint no-op."""
    _cd_to_tmp(monkeypatch, tmp_path)
    nibi = _MockBackend(kind="nibi")
    factory = _build_mock_factory(runpod=_MockBackend(kind="runpod"), nibi=nibi)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            ["launch", "--issue", "825", "--intent", "lora-7b", "--hydra", "smoke=1"],
            backends_factory=factory,
        )
    assert rc == 0
    assert not _lane_env_warnings(caplog)


def test_kill_switch_env_skips_lint_launch_proceeds(monkeypatch, tmp_path, caplog) -> None:
    """Plan test 16: EPM_SKIP_WORKLOAD_CMD_ENV_LINT=1 + the case-12 inputs →
    launch proceeds (kill switch), a single info line, no warning/flag."""
    _cd_to_tmp(monkeypatch, tmp_path)
    monkeypatch.setenv("EPM_SKIP_WORKLOAD_CMD_ENV_LINT", "1")
    caplog.set_level(logging.INFO, logger="dispatch_issue")
    import scripts.dispatch_issue as cli

    # Silence the orthogonal override-without-frontmatter warning path.
    monkeypatch.setattr(cli, "_frontmatter_backend_value", lambda _issue: "runpod")
    runpod = _MockBackend(kind="runpod")
    marker_posts: list[dict] = []
    factory = _build_mock_factory(runpod=runpod, marker_posts=marker_posts)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(
            [
                "launch",
                "--issue",
                "825",
                "--intent",
                "lora-7b",
                "--backend",
                "runpod",
                "--execute-workload",
                "--workload-cmd",
                INCIDENT_825_CMD,
            ],
            backends_factory=factory,
        )
    assert rc == 0
    assert len(runpod.launches) == 1
    assert not _lane_env_warnings(caplog)
    infos = [
        rec.getMessage()
        for rec in caplog.records
        if rec.levelno == logging.INFO and "EPM_SKIP_WORKLOAD_CMD_ENV_LINT" in rec.getMessage()
    ]
    assert len(infos) == 1
    for extra in _backend_selected_extras(marker_posts):
        assert "workload_cmd_lane_env_risk" not in extra


# ---------------------------------------------------------------------------
# backend_poll failover-time breadcrumb (plan §4 test 17)
# ---------------------------------------------------------------------------


def _gcp_handle_with_cmd(cmd: str) -> RunHandle:
    return RunHandle(
        backend="gcp",
        cluster=None,
        job_id="1234567890",
        pod_name="eps-issue-825",
        scratch_dir="/workspace",
        log_path="/workspace/logs/issue-825.log",
        extra={
            "intent": "lora-7b",
            "gpus": None,
            "time_budget_hours": None,
            "hydra_args": [],
            "workload_cmd": cmd,
            "repo_branch": "issue-825",
        },
    )


def test_runspec_from_gcp_handle_warns_and_never_alters_spec(monkeypatch, caplog) -> None:
    """Plan test 17 (+ critic 9): the failover spec-builder logs a warn-only
    breadcrumb naming the flagged var and returns a spec EQUAL to the one
    built with the lint killed via EPM_SKIP_WORKLOAD_CMD_ENV_LINT=1."""
    from scripts.backend_poll import _runspec_from_gcp_handle

    handle = _gcp_handle_with_cmd(INCIDENT_825_CMD)
    caplog.set_level(logging.WARNING)
    spec_linted = _runspec_from_gcp_handle(handle, 825)
    breadcrumbs = [
        rec.getMessage() for rec in caplog.records if "workload-cmd-lane-env" in rec.getMessage()
    ]
    assert breadcrumbs, "expected the failover-time lane-env breadcrumb"
    assert "WORKLOAD_ROOT" in breadcrumbs[0]

    caplog.clear()
    monkeypatch.setenv("EPM_SKIP_WORKLOAD_CMD_ENV_LINT", "1")
    spec_unlinted = _runspec_from_gcp_handle(handle, 825)
    assert not [rec for rec in caplog.records if "workload-cmd-lane-env" in rec.getMessage()]
    assert spec_linted == spec_unlinted


def test_runspec_from_gcp_handle_clean_cmd_no_breadcrumb(caplog) -> None:
    from scripts.backend_poll import _runspec_from_gcp_handle

    caplog.set_level(logging.WARNING)
    spec = _runspec_from_gcp_handle(_gcp_handle_with_cmd(FIXED_825_CMD), 825)
    assert spec.workload_cmd == FIXED_825_CMD
    assert not [rec for rec in caplog.records if "workload-cmd-lane-env" in rec.getMessage()]


# ---------------------------------------------------------------------------
# Parity pins (plan §4 test 18 + critic addition 4) and prose pin (test 19)
# ---------------------------------------------------------------------------

_RENDERER_SOURCES = {
    "gcp": REPO / "src" / "explore_persona_space" / "backends" / "gcp.py",
    "runpod": REPO / "src" / "explore_persona_space" / "backends" / "runpod.py",
    "slurm": REPO / "src" / "explore_persona_space" / "backends" / "slurm.py",
}

#: Ambient vars exempt from the FORWARD parity check: HOME/PATH sit in every
#: lane's export set because the executing shell inherits them ambiently —
#: they are NOT literal ``export VAR=`` lines in every renderer (runpod and
#: slurm inherit HOME from the SSH/root shell). Critic addition (1).
_AMBIENT_VARS = frozenset({"HOME", "PATH"})


def test_forward_parity_lane_exports_exist_in_renderer_sources() -> None:
    """Plan test 18: every non-ambient var in each lane's curated set appears
    as a literal ``export VAR=`` in that lane's renderer source — the curated
    map cannot silently go stale against a renderer edit.

    Known limitation (documented, accepted): a source-text grep cannot tell
    an unconditional export from one inside a conditional branch (e.g. gcp's
    REPO_ROOT export lives in the workload-cmd branch; slurm's EPS_* exports
    live in the custom-stage branch) — a var whose export moves to a branch
    the workload-cmd path does not take would false-PASS here. The lint's
    warn-by-default posture bounds the blast radius of that false pass.
    """
    for lane, path in _RENDERER_SOURCES.items():
        src = path.read_text(encoding="utf-8")
        for var in sorted(LANE_WORKLOAD_ENV_EXPORTS[lane] - _AMBIENT_VARS):
            assert re.search(rf"export {var}=", src), (
                f"{lane}: {var} is in LANE_WORKLOAD_ENV_EXPORTS but has no "
                f"'export {var}=' line in {path.name} — the curated map is stale"
            )


def test_inverse_parity_universe_scoped() -> None:
    """Critic addition (4): for each var in the candidate UNIVERSE only, if a
    lane's renderer source exports it, the var must be in that lane's set —
    prevents a stale map wrongly warning/refusing after a renderer later adds
    an export (the #641 REPO_ROOT precedent). Deliberately universe-scoped:
    renderers export noise vars (HF_XET_*, HF_HOME, ...) that must not join
    the lint universe."""
    universe = frozenset().union(*LANE_WORKLOAD_ENV_EXPORTS.values())
    for lane, path in _RENDERER_SOURCES.items():
        src = path.read_text(encoding="utf-8")
        for var in sorted(universe):
            if re.search(rf"export {var}=", src):
                assert var in LANE_WORKLOAD_ENV_EXPORTS[lane], (
                    f"{lane}: renderer {path.name} exports {var} but "
                    f"LANE_WORKLOAD_ENV_EXPORTS[{lane!r}] lacks it — the curated map "
                    "is stale and would wrongly warn/refuse"
                )


def test_skill_step6b_lane_portable_repo_root_pin() -> None:
    """Plan test 19 / §10 durability pin: SKILL.md Step 6b no longer
    prescribes the bare ``REPO_ROOT="$WORKLOAD_ROOT" bash`` composition and
    DOES carry the lane-portable ``${WORKLOAD_ROOT:-`` recommendation."""
    skill = (REPO / ".claude" / "skills" / "issue" / "SKILL.md").read_text(encoding="utf-8")
    assert 'REPO_ROOT="$WORKLOAD_ROOT" bash' not in skill, (
        "SKILL.md re-prescribes the bare $WORKLOAD_ROOT composition (#825/#1329)"
    )
    assert "${WORKLOAD_ROOT:-" in skill
