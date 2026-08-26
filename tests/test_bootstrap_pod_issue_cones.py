"""Pod sparse-cone selection (#1739 incident, 2026-08-04).

Root cause: both step-4 cone-set sites in ``scripts/bootstrap_pod.sh``
referenced ``\\${ISSUE:-}`` — backslash-escaped, i.e. expanded on the REMOTE
side of ``ssh_cmd`` — but ``ssh_cmd`` forwards no environment variables
(plain ``ssh host "cmd"``; sshd ``AcceptEnv`` defaults to LANG/LC_* only).
The remote test was therefore ALWAYS false and every pod came up with only
the then-default cone set (``configs docs scripts src tests``; tracked
``data/`` joined the defaults in #2211), regardless of the
``ISSUE`` env var ``pod_lifecycle.py::_bootstrap`` exported locally. Two
independent #1739 pods crashed FileNotFoundError on a git-tracked
``eval_results/issue_1739/...`` input the same day. A SECONDARY defect made
re-bootstraps unfixable even with the env seam closed: on the existing-repo
path the whole cone block was nested inside the
``if ! git config --get remote.origin.promisor`` retrofit guard, which is
false on every pod bootstrapped after #2051.

These tests are honest about the mechanism: they MATERIALIZE the step-4
ssh payload via real bash (``ssh_cmd`` stubbed to capture its argument, the
script's own local-capture lines included verbatim), then EXECUTE that
payload against a scratch ``file://`` origin under ``env -i`` — a clean
environment with no ``ISSUE`` — which faithfully reproduces the remote pod
shell. No re-implementation of bash quoting, no mocking of git or of the
cone-selection logic itself.
"""

from __future__ import annotations

import inspect
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap_pod.sh"

sys.path.insert(0, str(REPO_ROOT / "scripts"))

# The closing quote of the step-4 payload sits at column 0 (`\n"; then`);
# a bare '"; then' would false-match the block's own escaped `\"; then`
# sequences (e.g. `... origin \"\$BRANCH\"; then`).
_STEP4_OPEN = 'if ssh_cmd "\nset -eu\nBRANCH='
_STEP4_CLOSE = '\n"; then'


def _script_text() -> str:
    return BOOTSTRAP.read_text(encoding="utf-8")


def _step4_block() -> str:
    """The full ``if ssh_cmd "..."; then`` step-4 git-setup block."""
    text = _script_text()
    start = text.index(_STEP4_OPEN)
    end = text.index(_STEP4_CLOSE, start) + len(_STEP4_CLOSE)
    return text[start:end]


def _local_capture_lines() -> list[str]:
    """The script's own top-level ISSUE_VAL / EXTRA_CONES_VAL capture lines.

    Extracted verbatim so the test executes the script's actual mechanism;
    pre-fix (no capture lines) this list is empty and the payload keeps its
    broken remote-side ``${ISSUE:-}`` references.
    """
    return [
        line
        for line in _script_text().splitlines()
        if line.startswith(("ISSUE_VAL=", "EXTRA_CONES_VAL="))
    ]


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.fixture()
def scratch() -> Iterator[Path]:
    """A mkdtemp scratch root — deliberately NOT pytest's ``tmp_path``.

    ``tmp_path`` lives under the shared ``/tmp/pytest-of-<user>/pytest-<N>``
    numbered roots, and pytest prunes all but the newest 3 roots at session
    start — so on this shared VM, where many pytest sessions run
    concurrently, another session's startup can delete THIS session's live
    scratch repos mid-test (observed: 8/8 parallel pytest failures with the
    payload's cones reading code-only / rc!=0, while 8/8 parallel plain
    ``mkdtemp`` reproductions passed). ``mkdtemp`` roots are never pruned.
    """
    d = Path(tempfile.mkdtemp(prefix="bootstrap-pod-cones-"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def origin(scratch: Path) -> Path:
    """A scratch non-bare origin with code dirs + two issues' artifacts."""
    src = scratch / "origin"
    for d in (
        "src",
        "scripts",
        "configs",
        "tests",
        "docs",
        "data/assistant_axis",
        "eval_results/issue_1739/pvsynth",
        "figures/issue_1739",
        "eval_results/issue_400",
    ):
        (src / d).mkdir(parents=True)
        (src / d / "x.txt").write_text(d + "\n")
    _git("init", "-q", "-b", "main", cwd=src)
    _git("add", "-A", cwd=src)
    _git(
        "-c",
        "user.email=t@example.com",
        "-c",
        "user.name=t",
        "commit",
        "-q",
        "-m",
        "seed",
        cwd=src,
    )
    # Partial-clone support for the payload's --filter=blob:none fetches.
    _git("config", "uploadpack.allowFilter", "true", cwd=src)
    return src


def _materialize_payload(
    scratch: Path,
    origin: Path,
    *,
    issue: str | None = None,
    extra_cones: str | None = None,
) -> Path:
    """Run the extracted step-4 block through real bash with ssh_cmd stubbed.

    Returns the path of the captured payload — byte-identical to the command
    string the real bootstrap would send over ssh for the same local env.
    """
    payload = scratch / "payload.sh"
    lines = [
        "#!/bin/bash",
        "set -u",
        f'ssh_cmd() {{ printf %s "$1" > {shlex.quote(str(payload))}; }}',
        "log_ok() { :; }",
        "log_fail() { :; }",
        "BOOTSTRAP_BRANCH=main",
        f"REMOTE_DIR={shlex.quote(str(scratch / 'pod-repo'))}",
        f"REPO_URL_TOKENLESS={shlex.quote('file://' + str(origin))}",
        "GIT_CRED_HELPER='!f() { :; }; f'",
    ]
    if issue is not None:
        lines.append(f"export ISSUE={shlex.quote(issue)}")
    if extra_cones is not None:
        lines.append(f"export BOOTSTRAP_EXTRA_CONES={shlex.quote(extra_cones)}")
    lines += _local_capture_lines()
    lines.append(_step4_block() + "\n    :\nfi\n")
    driver = scratch / "driver.sh"
    driver.write_text("\n".join(lines))
    subprocess.run(["bash", str(driver)], check=True, capture_output=True, text=True)
    assert payload.exists(), "ssh_cmd stub never fired — step-4 block extraction broke"
    return payload


def _exec_payload_clean_env(payload: Path, scratch: Path) -> subprocess.CompletedProcess:
    """Execute the payload the way the pod would: NO local env leaks through.

    ``gc.auto=0`` (env-based config, git >= 2.31) pins out background
    ``git gc --auto`` maintenance: a first payload's fetch/reset can detach
    an auto-gc on the promisor+shallow scratch repo whose lingering locks
    race a second payload's ``git pull`` under host load (observed as a
    load-dependent rc!=0 flake with ~11 s runs vs ~3 s passes). The cone
    logic under test never depends on gc.
    """
    return subprocess.run(
        [
            "env",
            "-i",
            f"PATH={os.environ['PATH']}",
            f"HOME={scratch}",
            "GIT_CONFIG_NOSYSTEM=1",
            "GIT_CONFIG_COUNT=1",
            "GIT_CONFIG_KEY_0=gc.auto",
            "GIT_CONFIG_VALUE_0=0",
            "bash",
            str(payload),
        ],
        capture_output=True,
        text=True,
        cwd=str(scratch),
    )


def _cones(repo: Path) -> set[str]:
    r = subprocess.run(
        ["git", "-C", str(repo), "sparse-checkout", "list"],
        capture_output=True,
        text=True,
    )
    return set(r.stdout.split())


# ---------------------------------------------------------------------------
# 1. Fresh clone: the declared issue's cones must open (the #1739 crash path)
# ---------------------------------------------------------------------------


def test_fresh_clone_opens_declared_issue_cones(scratch: Path, origin: Path) -> None:
    payload = _materialize_payload(scratch, origin, issue="1739")
    result = _exec_payload_clean_env(payload, scratch)
    assert result.returncode == 0, f"payload failed:\n{result.stdout}\n{result.stderr}"
    repo = scratch / "pod-repo"
    cones = _cones(repo)
    assert "eval_results/issue_1739" in cones, (
        f"issue cone missing — the pod would crash FileNotFoundError on its own "
        f"tracked inputs (realized cones: {sorted(cones)})"
    )
    assert "figures/issue_1739" in cones
    # The tracked artifact actually materialized (not just a cone entry).
    assert (repo / "eval_results/issue_1739/pvsynth/x.txt").is_file()
    assert (repo / "src/x.txt").is_file(), "code cones must still open"
    # Tracked data/ inputs are in the default cone set (#2211; the #2203
    # Phase 3 crash path: data/assistant_axis/role_list.json).
    assert (repo / "data/assistant_axis/x.txt").is_file()
    # Still sparse: an UNdeclared issue's artifacts stay out.
    assert not (repo / "eval_results/issue_400").exists()
    assert "WARNING" not in result.stderr


def test_fresh_clone_without_issue_opens_default_cones(scratch: Path, origin: Path) -> None:
    """No issue declared: default cones (code + tracked data/) open, issue cones stay out."""
    payload = _materialize_payload(scratch, origin)
    result = _exec_payload_clean_env(payload, scratch)
    assert result.returncode == 0, f"payload failed:\n{result.stdout}\n{result.stderr}"
    repo = scratch / "pod-repo"
    assert (repo / "src/x.txt").is_file()
    # Tracked data/ is part of the default cone set as of #2211.
    assert (repo / "data/assistant_axis/x.txt").is_file()
    assert not (repo / "eval_results").exists()
    # No issue declared => no missing-cone warning.
    assert "WARNING" not in result.stderr


# ---------------------------------------------------------------------------
# 2. Re-bootstrap (existing repo, promisor already configured): the issue
#    cones must STILL open — the promisor-retrofit guard must not gate them.
# ---------------------------------------------------------------------------


def test_rebootstrap_with_promisor_configured_opens_issue_cones(
    scratch: Path, origin: Path
) -> None:
    # First bootstrap, no issue: creates the repo with promisor + default cones.
    first = _materialize_payload(scratch, origin)
    r1 = _exec_payload_clean_env(first, scratch)
    assert r1.returncode == 0, f"first bootstrap failed:\n{r1.stdout}\n{r1.stderr}"
    repo = scratch / "pod-repo"
    assert "eval_results/issue_1739" not in _cones(repo)

    # Simulate the PRE-#2211 cone state (every pod bootstrapped before the
    # 'data' cone joined the defaults): reset to the old data-less default
    # set. Without this reset the first (post-fix) payload already installed
    # the 'data' cone, and the data assertion after the second payload would
    # pass VACUOUSLY without ever exercising the always-running
    # `sparse-checkout add data` on the existing-repo path.
    _git("sparse-checkout", "set", "src", "scripts", "configs", "tests", "docs", cwd=repo)
    assert not (repo / "data").exists(), "pre-fix simulation must drop the data cone"

    # Re-bootstrap the SAME repo with an issue declared (the hand-patch case
    # from 2026-08-04: pod exists, promisor configured, retrofit guard false).
    second = _materialize_payload(scratch, origin, issue="1739")
    r2 = _exec_payload_clean_env(second, scratch)
    assert r2.returncode == 0, f"re-bootstrap failed:\n{r2.stdout}\n{r2.stderr}"
    cones = _cones(repo)
    assert "eval_results/issue_1739" in cones, (
        f"re-bootstrap must open the issue cones even when the promisor "
        f"retrofit guard is false (realized cones: {sorted(cones)})"
    )
    assert (repo / "eval_results/issue_1739/pvsynth/x.txt").is_file()
    # The always-running add restored the default 'data' cone on an existing
    # sparse repo whose cone set predates #2211.
    assert (repo / "data/assistant_axis/x.txt").is_file(), (
        f"re-bootstrap must add the default 'data' cone to a pre-#2211 sparse "
        f"repo (realized cones: {sorted(_cones(repo))})"
    )
    assert "WARNING" not in r2.stderr


# ---------------------------------------------------------------------------
# 3. Cross-issue artifact reuse: BOOTSTRAP_EXTRA_CONES opens foreign cones
# ---------------------------------------------------------------------------


def test_extra_cones_env_opens_foreign_issue_cones(scratch: Path, origin: Path) -> None:
    payload = _materialize_payload(
        scratch, origin, issue="1739", extra_cones="eval_results/issue_400"
    )
    result = _exec_payload_clean_env(payload, scratch)
    assert result.returncode == 0, f"payload failed:\n{result.stdout}\n{result.stderr}"
    repo = scratch / "pod-repo"
    cones = _cones(repo)
    assert "eval_results/issue_400" in cones, f"extra cone missing: {sorted(cones)}"
    assert (repo / "eval_results/issue_400/x.txt").is_file()
    assert "eval_results/issue_1739" in cones
    # Leg A negative (#2608): every declared extra cone landed => no WARNING.
    assert "WARNING" not in result.stderr


# ---------------------------------------------------------------------------
# 4. Static pins on the seam itself
# ---------------------------------------------------------------------------


def test_no_remote_side_issue_reference_in_bootstrap() -> None:
    """The root-cause shape must stay dead: no escaped (remote-side) ${ISSUE}.

    ``ssh_cmd`` forwards no env vars, so any ``\\${ISSUE`` inside the
    double-quoted payload reads empty on the pod — silently.
    """
    assert "\\${ISSUE" not in _script_text(), (
        "remote-side ${ISSUE} reference reintroduced in bootstrap_pod.sh — "
        "ssh forwards no env vars; capture locally (ISSUE_VAL) instead"
    )


def test_step4_echoes_cones_and_warns_on_missing_issue_cone() -> None:
    block = _step4_block()
    assert "Sparse cones:" in block, "step 4 must echo the realized cone list"
    assert "WARNING" in block and "git sparse-checkout add eval_results/issue_" in block, (
        "step 4 must warn loudly, naming the one-line remedy, when a declared "
        "issue's cones did not land"
    )
    assert "git sparse-checkout add data" in block.split("Sparse cones:", 1)[1], (
        "step 4's cone verification must warn, naming the one-line remedy, "
        "when the default 'data' cone did not land (#2211)"
    )


def test_every_sparse_checkout_set_site_includes_data_cone() -> None:
    """All three cone-SET sites carry the default 'data' cone (#2211).

    Sites: legacy promisor-retrofit, fresh-clone with issue, fresh-clone
    without issue. A set site that drops 'data' silently reverts the #2203
    Phase 3 crash class (tracked data/ inputs absent on a fresh pod).
    """
    set_lines = [ln for ln in _script_text().splitlines() if "git sparse-checkout set" in ln]
    assert len(set_lines) == 3, f"expected the 3 known cone-set sites, got: {set_lines}"
    for ln in set_lines:
        args = ln.split("git sparse-checkout set", 1)[1]
        assert re.search(r"\bdata\b", args), f"cone-set site missing the 'data' token: {ln!r}"


def test_always_running_data_cone_add_outside_retrofit_guard() -> None:
    """The existing-repo path adds 'data' UNCONDITIONALLY (#2211, #1739 lesson).

    The promisor-retrofit guard is false on every post-#2051 pod, so an add
    nested inside it would never reach an EXISTING sparse repo whose cone set
    predates #2211 — the add must live in the always-running
    ``core.sparseCheckout = true`` block, before the per-issue add.
    """
    block = _step4_block()
    retrofit_start = block.index("if ! git config --get remote.origin.promisor")
    # Anchor on the guard's own indented close — a bare "fi" would false-match
    # the "fi" inside "partialclonefilter" on the guard's second line.
    retrofit_end = block.index("\n    fi\n", retrofit_start)
    assert "sparse-checkout add data" not in block[retrofit_start:retrofit_end], (
        "the default 'data' cone add must NOT be nested inside the "
        "promisor-retrofit guard (false on every post-#2051 pod)"
    )
    sparse_block_start = block.index("core.sparseCheckout", retrofit_end)
    add_idx = block.index("git sparse-checkout add data")
    per_issue_add_idx = block.index('git sparse-checkout add \\"eval_results/issue_$ISSUE_VAL\\"')
    assert sparse_block_start < add_idx < per_issue_add_idx, (
        "the 'add data' line must sit inside the always-running sparse block, "
        "before the per-issue add"
    )


def test_rebootstrap_test_simulates_pre2211_dataless_cone_set() -> None:
    """Pin the re-bootstrap test's pre-fix simulation (guards against vacuity).

    The re-bootstrap test must perform a data-less ``sparse-checkout set``
    BEFORE its second payload execution — otherwise the first (post-fix)
    payload already installs the 'data' cone and the data assertion passes
    without exercising the always-running add (critic round-1 finding).
    """
    src = inspect.getsource(test_rebootstrap_with_promisor_configured_opens_issue_cones)
    dataless_set = '_git("sparse-checkout", "set", "src", "scripts", "configs", "tests", "docs"'
    assert dataless_set in src, (
        "the re-bootstrap test must reset the pod repo to the pre-#2211 "
        "data-less default cone set between its two payload executions"
    )
    assert src.index(dataless_set) < src.index("second = _materialize_payload"), (
        "the pre-fix simulation must run BEFORE the second payload executes"
    )


# ---------------------------------------------------------------------------
# 5. pod.py bootstrap derives ISSUE from the pod name (manual re-bootstraps)
# ---------------------------------------------------------------------------


def test_pod_py_derives_issue_from_pod_name() -> None:
    import pod

    assert pod._derive_issue_from_pod_name("pod-1739") == "1739"
    assert pod._derive_issue_from_pod_name("pod-1739-r2fair") == "1739"
    assert pod._derive_issue_from_pod_name("epm-issue-42") == "42"
    assert pod._derive_issue_from_pod_name("epm-issue-42-b") == "42"
    assert pod._derive_issue_from_pod_name("pod1") is None  # permanent pod
    assert pod._derive_issue_from_pod_name("pod-foo") is None
    assert pod._derive_issue_from_pod_name(None) is None


def test_pod_py_bootstrap_env_carries_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    import pod

    import explore_persona_space.task_workflow as tw

    # Hermetic: keep the (fail-soft, #2608 r3) cone derivation off the REAL
    # task registry — this test pins ISSUE precedence only.
    def _no_task(task_id: int) -> Path:
        raise FileNotFoundError(f"task #{task_id} (synthetic)")

    monkeypatch.setattr(tw, "find_task_path", _no_task)
    monkeypatch.delenv("ISSUE", raising=False)
    env = pod._bootstrap_env_with_intent("pod-1739-r2fair")
    assert env.get("ISSUE") == "1739"

    # Explicit operator override always wins.
    monkeypatch.setenv("ISSUE", "77")
    env = pod._bootstrap_env_with_intent("pod-1739")
    assert env["ISSUE"] == "77"

    # No pod name / underivable => ISSUE stays unset (default cones only).
    monkeypatch.delenv("ISSUE", raising=False)
    env = pod._bootstrap_env_with_intent(None)
    assert "ISSUE" not in env


# ---------------------------------------------------------------------------
# 5b. pod.py MANUAL bootstrap path derives plan cones (#2608 round 3 —
#     concern pod-py-bootstrap-path-no-derivation). Both public bootstrap
#     paths share pod_lifecycle.merge_derived_extra_cones, so a
#     `provision --no-bootstrap` followed by the documented
#     `pod.py bootstrap <name>` recovery opens the same cross-issue cones.
# ---------------------------------------------------------------------------


def _plan_task_dir(tmp_path: Path, issue: int, plan_text: str) -> Path:
    """A tasks/<status>/<N>-shaped dir holding one persisted plans/v1.md."""
    task_dir = tmp_path / "running" / str(issue)
    (task_dir / "plans").mkdir(parents=True)
    (task_dir / "plans" / "v1.md").write_text(plan_text)
    return task_dir


def test_pod_py_manual_bootstrap_env_derives_cones_without_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Real _bootstrap_env_with_intent body: with NO caller export, the manual
    path derives the plan-cited foreign cones (pre-r3 it set only
    POD_INTENT + ISSUE). Boundary fake: task_workflow.find_task_path only —
    the merge helper + extra_cones_for_plan chain runs for real."""
    import pod

    import explore_persona_space.task_workflow as tw

    task_dir = _plan_task_dir(
        tmp_path,
        1739,
        "Reads eval_results/issue_1482/x.json and its own eval_results/issue_1739/own.json\n",
    )
    monkeypatch.setattr(tw, "find_task_path", lambda task_id: task_dir)
    monkeypatch.setenv("POD_INTENT", "eval")  # precedence-1 branch: no sidecar read
    monkeypatch.delenv("ISSUE", raising=False)
    monkeypatch.delenv("BOOTSTRAP_EXTRA_CONES", raising=False)
    env = pod._bootstrap_env_with_intent("pod-1739-r2fair")
    assert env["ISSUE"] == "1739"
    assert env["BOOTSTRAP_EXTRA_CONES"] == "eval_results/issue_1482"
    assert "Extra sparse cones" in capsys.readouterr().out


def test_pod_py_manual_bootstrap_env_unions_caller_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caller export survives FIRST; derived cones union in (order-stable)."""
    import pod

    import explore_persona_space.task_workflow as tw

    task_dir = _plan_task_dir(tmp_path, 1739, "Reads figures/issue_2476/z.png\n")
    monkeypatch.setattr(tw, "find_task_path", lambda task_id: task_dir)
    monkeypatch.setenv("POD_INTENT", "eval")
    monkeypatch.delenv("ISSUE", raising=False)
    monkeypatch.setenv("BOOTSTRAP_EXTRA_CONES", "eval_results/issue_7")
    env = pod._bootstrap_env_with_intent("pod-1739")
    assert env["BOOTSTRAP_EXTRA_CONES"] == "eval_results/issue_7 figures/issue_2476"


def test_pod_py_manual_bootstrap_env_fail_soft_leaves_env_untouched(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail-soft pin: a derivation error never raises, never mutates the
    caller env, and surfaces its one-line stderr note (never a silent pass)."""
    import pod

    import explore_persona_space.task_workflow as tw

    def _boom(task_id: int) -> Path:
        raise RuntimeError("registry corrupted (synthetic)")

    monkeypatch.setattr(tw, "find_task_path", _boom)
    monkeypatch.setenv("POD_INTENT", "eval")
    monkeypatch.delenv("ISSUE", raising=False)
    monkeypatch.delenv("BOOTSTRAP_EXTRA_CONES", raising=False)
    env = pod._bootstrap_env_with_intent("pod-42")
    assert env["ISSUE"] == "42"
    assert "BOOTSTRAP_EXTRA_CONES" not in env
    err = capsys.readouterr().err
    assert "extra-cone derivation skipped for issue 42" in err
    assert "RuntimeError" in err


def test_pod_py_manual_bootstrap_env_non_numeric_issue_notes_and_skips(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed operator ISSUE export fail-softs at the wrapper's own
    int() with a stderr note — env untouched, bootstrap proceeds."""
    import pod

    import explore_persona_space.task_workflow as tw

    monkeypatch.setattr(
        tw, "find_task_path", lambda task_id: pytest.fail("must not resolve a task")
    )
    monkeypatch.setenv("POD_INTENT", "eval")
    monkeypatch.setenv("ISSUE", "not-a-number")
    monkeypatch.delenv("BOOTSTRAP_EXTRA_CONES", raising=False)
    env = pod._bootstrap_env_with_intent("pod-42")
    assert env["ISSUE"] == "not-a-number"  # explicit override preserved verbatim
    assert "BOOTSTRAP_EXTRA_CONES" not in env
    err = capsys.readouterr().err
    assert "extra-cone derivation skipped for manual bootstrap" in err
    assert "ValueError" in err


def test_pod_py_manual_bootstrap_no_issue_no_derivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No resolvable issue => derivation never attempted (negative control:
    a task resolution would fail the test)."""
    import pod

    import explore_persona_space.task_workflow as tw

    monkeypatch.setattr(
        tw, "find_task_path", lambda task_id: pytest.fail("must not resolve a task")
    )
    monkeypatch.setenv("POD_INTENT", "eval")
    monkeypatch.delenv("ISSUE", raising=False)
    monkeypatch.delenv("BOOTSTRAP_EXTRA_CONES", raising=False)
    env = pod._bootstrap_env_with_intent(None)
    assert "ISSUE" not in env
    assert "BOOTSTRAP_EXTRA_CONES" not in env


def test_pod_py_manual_path_routes_through_shared_merge_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC3 choke-point pin (dispatch-level): the manual path calls the SAME
    shared helper provision's _bootstrap_env uses. Body coverage of the real
    helper lives in the sibling tests above + tests/test_pod_lifecycle.py."""
    import pod
    import pod_lifecycle

    calls: list[tuple[int, str | None]] = []

    def _recorder(env: dict[str, str], issue: int) -> dict[str, str]:
        calls.append((issue, env.get("BOOTSTRAP_EXTRA_CONES")))
        return env

    monkeypatch.setattr(pod_lifecycle, "merge_derived_extra_cones", _recorder)
    monkeypatch.setenv("POD_INTENT", "eval")
    monkeypatch.delenv("ISSUE", raising=False)
    monkeypatch.delenv("BOOTSTRAP_EXTRA_CONES", raising=False)
    pod._bootstrap_env_with_intent("pod-77")
    assert calls == [(77, None)]


# ---------------------------------------------------------------------------
# 6. Post-checkout audit legs A + B (#2608): declared-cone verification and
#    the driver-grep backstop for undeclared cross-issue reads. EXECUTED
#    payload tests — anchor-only presence checks cannot tell a never-firing
#    warning from a firing one (the incident shape one level up).
# ---------------------------------------------------------------------------


def test_audit_leg_a_warns_on_declared_but_unrealized_extra_cone(
    scratch: Path, origin: Path
) -> None:
    """Leg A positive: a DECLARED extra cone that never landed draws a WARNING.

    ``git sparse-checkout add`` of a not-yet-committed dir still lists it
    (probed at implementation time), so a healthy payload cannot realize the
    missing state; the regression this leg guards is the ADD not landing
    (a future refactor regressing the add sites — the #2569 shape one level
    up). Simulate it by no-op'ing the extra-cone add commands in the
    materialized payload, then execute: the audit must WARN with the remedy
    while rc stays 0 (warn loud, never hard-fail).
    """
    payload = _materialize_payload(
        scratch, origin, issue="1739", extra_cones="eval_results/issue_400"
    )
    text = payload.read_text()
    assert "git sparse-checkout add eval_results/issue_400" in text
    payload.write_text(
        text.replace("git sparse-checkout add eval_results/issue_400", ": regression-sim")
    )
    result = _exec_payload_clean_env(payload, scratch)
    assert result.returncode == 0, f"audit must never hard-fail:\n{result.stdout}\n{result.stderr}"
    assert "WARNING: declared extra cone eval_results/issue_400 MISSING" in result.stderr
    assert "git sparse-checkout add eval_results/issue_400" in result.stderr, (
        "the leg-A warning must name the one-line sparse-checkout add remedy"
    )


def test_audit_leg_b_warns_on_driver_referenced_foreign_cone(scratch: Path, origin: Path) -> None:
    """Leg B positive: a driver citing a foreign issue's artifacts with the
    cone unopened draws a WARNING; own-issue refs (incl. ood_) never warn."""
    driver = origin / "scripts" / "issue1739_reader.py"
    driver.write_text(
        'FOREIGN = "eval_results/issue_400/pvsynth/x.txt"\n'
        'OWN = "eval_results/issue_1739/pvsynth/x.txt"\n'
        # Own-issue ood_ ref: NOT in the default per-issue cone set, so only
        # the leg-B own-issue `continue` keeps it warning-free.
        'OWN_OOD = "ood_eval_results/issue_1739/raw.json"\n'
    )
    _git("add", "scripts/issue1739_reader.py", cwd=origin)
    _git(
        "-c",
        "user.email=t@example.com",
        "-c",
        "user.name=t",
        "commit",
        "-q",
        "-m",
        "driver",
        cwd=origin,
    )
    payload = _materialize_payload(scratch, origin, issue="1739")
    result = _exec_payload_clean_env(payload, scratch)
    assert result.returncode == 0, f"audit must never hard-fail:\n{result.stdout}\n{result.stderr}"
    assert "WARNING: driver-referenced foreign cone eval_results/issue_400 MISSING" in (
        result.stderr
    )
    assert "git sparse-checkout add eval_results/issue_400" in result.stderr, (
        "the leg-B warning must name the one-line sparse-checkout add remedy"
    )
    assert "issue_1739 MISSING" not in result.stderr, (
        "own-issue driver refs (incl. ood_eval_results/issue_1739) must not warn"
    )


def test_audit_leg_b_foreign_cone_present_no_warning(scratch: Path, origin: Path) -> None:
    """Leg B negative: the driver-referenced foreign cone IS open => no WARNING."""
    driver = origin / "scripts" / "issue1739_reader.py"
    driver.write_text('FOREIGN = "eval_results/issue_400/x.txt"\n')
    _git("add", "scripts/issue1739_reader.py", cwd=origin)
    _git(
        "-c",
        "user.email=t@example.com",
        "-c",
        "user.name=t",
        "commit",
        "-q",
        "-m",
        "driver",
        cwd=origin,
    )
    payload = _materialize_payload(
        scratch, origin, issue="1739", extra_cones="eval_results/issue_400"
    )
    result = _exec_payload_clean_env(payload, scratch)
    assert result.returncode == 0, f"payload failed:\n{result.stdout}\n{result.stderr}"
    assert "WARNING" not in result.stderr


def test_audit_leg_b_empty_driver_glob_is_inert(scratch: Path, origin: Path) -> None:
    """Leg B negative + empty-glob guard: no per-issue drivers => rc 0, no error.

    The origin fixture ships no ``scripts/issue1739_*.py``, so the unmatched
    glob must stay inert under the payload's ``set -eu`` (the ls guard), with
    no warning and no crash.
    """
    payload = _materialize_payload(scratch, origin, issue="1739")
    result = _exec_payload_clean_env(payload, scratch)
    assert result.returncode == 0, f"payload failed:\n{result.stdout}\n{result.stderr}"
    assert "WARNING" not in result.stderr
