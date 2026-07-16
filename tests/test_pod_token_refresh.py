"""Tests for ``pod.py keys --refresh-token`` (#1401).

The refresh leg (``scripts/sync_env_keys.sh --refresh-token``) is the
one-command pod git-auth repair + verification: re-push the VM ``.env``,
converge the pod git config to the #1239 contract (tokenless remote +
host-scoped env-reading credential helper + legacy-store scrub), then verify
with DUAL probes — an anonymous fetch-direction ``ls-remote origin HEAD``
AND an authenticated ``git push --dry-run origin HEAD:refs/heads/eps-auth-probe``
— classifying failures (egress-block / invalid token / inconclusive) with
no retry loop.

These tests are hermetic (no live pod, no network): ``ssh``/``scp`` are
stubbed via a PATH-prefixed fake-bin dir whose scripts append their full
argv to a log file and emit canned stdout / exit codes keyed on
remote-command substrings, controlled by ``FAKE_*`` env vars. The script's
pods.conf and local ``.env`` are pointed at tmp files via the
``EPS_PODS_CONF_OVERRIDE`` / ``EPS_LOCAL_ENV_OVERRIDE`` seams. Static pins
cover the single-definition invariant of the shared credential-helper lib
(``scripts/_git_cred_helper.sh``), the ``--dry-run`` flag, the
header-via-stdin API probe, and the ``sync env`` redirect hint.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
SYNC_ENV_KEYS = SCRIPTS / "sync_env_keys.sh"
SYNC_ENV = SCRIPTS / "sync_env.sh"
BOOTSTRAP = SCRIPTS / "bootstrap_pod.sh"
CRED_LIB = SCRIPTS / "_git_cred_helper.sh"

# Fake secret distributed via the tmp .env. The no-leak test asserts this
# literal never reaches any argv or the script's stdout/stderr.
SECRET = "sekret-token-do-not-leak"
REMOTE_DIR = "/workspace/explore-persona-space"

# Argv log record separators (newline-safe: remote command strings contain
# embedded newlines, so records are delimiter-framed, not line-framed).
_CALL = "<<<CALL "
_ARG = "<<<ARG>>>"
_END = "<<<END>>>"

_LOG_SNIPPET = (
    "{ printf '<<<CALL %s>>>' \"$TOOL\"; "
    'for a in "$@"; do printf \'<<<ARG>>>%s\' "$a"; done; '
    "printf '<<<END>>>\\n'; } >> \"$FAKE_LOG\"\n"
)

_SSH_STUB = (
    "#!/bin/bash\n"
    "# Hermetic ssh stub (#1401 tests): records argv, emits canned output\n"
    "# keyed on remote-command substrings + FAKE_* env vars.\n"
    "TOOL=ssh\n" + _LOG_SNIPPET + 'last="${@: -1}"\n'
    'case "$last" in\n'
    '  *"ls-remote origin HEAD"*)\n'
    "    printf '%s\\n' \"${FAKE_FETCH_OUT:-0123abc\tHEAD}\"\n"
    '    exit "${FAKE_FETCH_RC:-0}"\n'
    "    ;;\n"
    '  *"push --dry-run"*)\n'
    "    printf '%s\\n' \"${FAKE_PUSH_OUT:-Everything up-to-date}\"\n"
    '    exit "${FAKE_PUSH_RC:-0}"\n'
    "    ;;\n"
    "  *api.github.com*)\n"
    "    printf '%s' \"${FAKE_API_CODE:-200}\"\n"
    "    exit 0\n"
    "    ;;\n"
    '  *"grep -cP"*)\n'
    "    printf '9\\n'\n"
    "    exit 0\n"
    "    ;;\n"
    "esac\n"
    "exit 0\n"
)

_SCP_STUB = (
    "#!/bin/bash\n"
    "# Hermetic scp stub (#1401 tests): records argv, succeeds.\n"
    "TOOL=scp\n" + _LOG_SNIPPET + "exit 0\n"
)


@dataclass
class RefreshHarness:
    env: dict[str, str]
    log: Path
    env_file: Path
    conf: Path


@pytest.fixture()
def harness(tmp_path: Path) -> RefreshHarness:
    """PATH-prefixed fake ssh/scp + tmp pods.conf + tmp .env with the secret."""
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    for name, body in (("ssh", _SSH_STUB), ("scp", _SCP_STUB)):
        stub = fakebin / name
        stub.write_text(body, encoding="utf-8")
        stub.chmod(0o755)
    log = tmp_path / "calls.log"
    log.touch()
    conf = tmp_path / "pods.conf"
    conf.write_text("pod-test 198.51.100.7 41234 1 H100 test\n", encoding="utf-8")
    env_file = tmp_path / "vm.env"
    env_file.write_text(f"GITHUB_TOKEN={SECRET}\nHF_TOKEN=unrelated\n", encoding="utf-8")
    env = os.environ.copy()
    for stale in (
        "FAKE_FETCH_RC",
        "FAKE_FETCH_OUT",
        "FAKE_PUSH_RC",
        "FAKE_PUSH_OUT",
        "FAKE_API_CODE",
    ):
        env.pop(stale, None)
    env.update(
        {
            "PATH": f"{fakebin}:{env['PATH']}",
            "FAKE_LOG": str(log),
            "EPS_PODS_CONF_OVERRIDE": str(conf),
            "EPS_LOCAL_ENV_OVERRIDE": str(env_file),
        }
    )
    return RefreshHarness(env=env, log=log, env_file=env_file, conf=conf)


def _run_refresh(env: dict[str, str]) -> subprocess.CompletedProcess:
    """Run the refresh leg against the stubbed harness (real bash, no network)."""
    return subprocess.run(
        ["bash", str(SYNC_ENV_KEYS), "--refresh-token", "pod-test"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def _calls(log: Path) -> list[tuple[str, list[str]]]:
    """Parse the fake-bin argv log into (tool, argv) records."""
    records: list[tuple[str, list[str]]] = []
    for rec in log.read_text(encoding="utf-8").split(_CALL)[1:]:
        head, _, rest = rec.partition(">>>")
        body = rest.split(_END)[0]
        args = body.split(_ARG)[1:]
        records.append((head, args))
    return records


def _remote_commands(log: Path) -> list[str]:
    """The remote command string (last argv element) of every recorded ssh call."""
    return [args[-1] for tool, args in _calls(log) if tool == "ssh" and args]


def _index_of(remotes: list[str], substr: str, start: int = 0) -> int:
    for i in range(start, len(remotes)):
        if substr in remotes[i]:
            return i
    pytest.fail(f"no remote command containing {substr!r} at index >= {start}: {remotes!r}")


def _materialized_helper() -> str:
    """The lib's GIT_CRED_HELPER value, materialized by real bash (no re-implementation)."""
    out = subprocess.run(
        [
            "bash",
            "-c",
            f"REMOTE_DIR={REMOTE_DIR}; source {shlex.quote(str(CRED_LIB))}; "
            'printf %s "$GIT_CRED_HELPER"',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


# ---------------------------------------------------------------------------
# 1. Happy path: full repair sequence + dual probes + VERIFIED
# ---------------------------------------------------------------------------


def test_refresh_composes_helper_string(harness: RefreshHarness) -> None:
    res = _run_refresh(harness.env)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    calls = _calls(harness.log)
    assert calls and calls[0][0] == "scp", f"first recorded call must be the .env scp: {calls!r}"
    assert str(harness.env_file) in calls[0][1], ".env push must reference the tmp .env PATH"

    remotes = _remote_commands(harness.log)
    i_seturl = _index_of(remotes, "remote set-url origin")
    assert "https://github.com/superkaiba/explore-persona-space.git" in remotes[i_seturl]
    assert "2>/dev/null" in remotes[i_seturl], "set-url first attempt mirrors bootstrap step 4"

    i_local = _index_of(
        remotes,
        f"git -C {REMOTE_DIR} config --replace-all credential.https://github.com.helper",
        i_seturl + 1,
    )
    helper_cmd = remotes[i_local]
    assert "username=x-access-token" in helper_cmd
    assert f"{REMOTE_DIR}/.env" in helper_cmd, "helper must carry the REMOTE_DIR-expanded .env"
    materialized = _materialized_helper()
    assert "'" not in materialized, "quoting invariant: no single quote in the helper value"
    assert f"'{materialized}'" in helper_cmd, (
        "the remote config argv must carry the lib's materialized helper inside "
        "remote-level single quotes"
    )
    assert "--unset-all credential.helper" in helper_cmd, "repo-local legacy-helper scrub"

    i_global = _index_of(
        remotes,
        "git config --global --replace-all credential.https://github.com.helper",
        i_local + 1,
    )
    assert "--unset-all credential.helper" in remotes[i_global]
    assert "rm -f /root/.git-credentials" in remotes[i_global], "legacy plaintext-store scrub"
    assert f"'{materialized}'" in remotes[i_global]

    i_fetch = _index_of(remotes, "ls-remote origin HEAD", i_global + 1)
    _index_of(remotes, "push --dry-run origin HEAD:refs/heads/eps-auth-probe", i_fetch + 1)
    assert "git auth VERIFIED" in res.stdout


# ---------------------------------------------------------------------------
# 2. Secrets hygiene: the token value never reaches argv or stdout/stderr
# ---------------------------------------------------------------------------


def test_refresh_no_token_leak(harness: RefreshHarness) -> None:
    res = _run_refresh(harness.env)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    log_text = harness.log.read_text(encoding="utf-8")
    assert SECRET not in log_text, "token literal leaked into an ssh/scp argv"
    assert SECRET not in res.stdout, "token literal leaked into stdout"
    assert SECRET not in res.stderr, "token literal leaked into stderr"


# ---------------------------------------------------------------------------
# 3. Working-tree safety: no checkout/pull/fetch, every push is --dry-run
# ---------------------------------------------------------------------------


def test_refresh_never_mutates_working_tree(harness: RefreshHarness) -> None:
    res = _run_refresh(harness.env)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    remotes = _remote_commands(harness.log)
    assert remotes, "expected recorded ssh remote commands"
    for cmd in remotes:
        assert not re.search(r"\bcheckout\b", cmd), f"working-tree mutation (checkout): {cmd!r}"
        assert not re.search(r"\bpull\b", cmd), f"working-tree mutation (pull): {cmd!r}"
        assert not re.search(r"\bfetch\b", cmd), f"working-tree mutation (fetch): {cmd!r}"
        for m in re.finditer(r"\bpush\b", cmd):
            tail = cmd[m.end() : m.end() + len(" --dry-run")]
            assert tail == " --dry-run", f"non-dry-run push in remote command: {cmd!r}"


# ---------------------------------------------------------------------------
# 4/5. Failure classification: push 40x with valid vs invalid token
# ---------------------------------------------------------------------------

_PUSH_403 = (
    "fatal: unable to access "
    "'https://github.com/superkaiba/explore-persona-space.git/': "
    "The requested URL returned error: 403"
)


def test_refresh_probe_403_valid_token_points_at_sideload(harness: RefreshHarness) -> None:
    harness.env.update({"FAKE_PUSH_RC": "128", "FAKE_PUSH_OUT": _PUSH_403, "FAKE_API_CODE": "200"})
    res = _run_refresh(harness.env)
    assert res.returncode != 0
    assert "#1315-r10 class" in res.stdout, "must name the valid-token 40x class"
    assert "contents:write" in res.stdout, "must name the scope-deficient-PAT possibility"
    assert "gotchas.md" in res.stdout and "bundle-sideload" in res.stdout
    remotes = _remote_commands(harness.log)
    n_push = sum("push --dry-run" in cmd for cmd in remotes)
    assert n_push == 1, f"exactly ONE push-probe attempt (no retry loop), got {n_push}"


def test_refresh_probe_403_invalid_token_says_rotate(harness: RefreshHarness) -> None:
    harness.env.update({"FAKE_PUSH_RC": "128", "FAKE_PUSH_OUT": _PUSH_403, "FAKE_API_CODE": "401"})
    res = _run_refresh(harness.env)
    assert res.returncode != 0
    assert "Rotate GITHUB_TOKEN" in res.stdout
    assert str(harness.env_file) in res.stdout, "rotate instruction must name the VM .env path"


# ---------------------------------------------------------------------------
# 5b. Fetch-direction gate: anonymous 40x is NOT a token problem
# ---------------------------------------------------------------------------


def test_fetch_probe_gate(harness: RefreshHarness) -> None:
    harness.env.update({"FAKE_FETCH_RC": "128", "FAKE_FETCH_OUT": _PUSH_403})
    res = _run_refresh(harness.env)
    assert res.returncode != 0
    assert "NOT a token problem" in res.stdout
    assert "gotchas.md" in res.stdout and "bundle-sideload" in res.stdout
    assert "authenticated push probe PASSed" in res.stdout
    assert "git auth VERIFIED" not in res.stdout


# ---------------------------------------------------------------------------
# 5c. API discriminator itself fails: INCONCLUSIVE, never a rotate instruction
# ---------------------------------------------------------------------------


def test_api_probe_inconclusive_branch(harness: RefreshHarness) -> None:
    harness.env.update(
        {"FAKE_PUSH_RC": "128", "FAKE_PUSH_OUT": _PUSH_403, "FAKE_API_CODE": "api-probe-failed"}
    )
    res = _run_refresh(harness.env)
    assert res.returncode != 0
    assert "probe INCONCLUSIVE" in res.stdout
    assert "Rotate GITHUB_TOKEN" not in res.stdout, (
        "a failed discriminator is NOT evidence the token is bad"
    )


# ---------------------------------------------------------------------------
# 5d. Static pin: API-probe auth header rides stdin, never curl argv
# ---------------------------------------------------------------------------


def test_api_probe_header_via_stdin_pinned() -> None:
    text = SYNC_ENV_KEYS.read_text(encoding="utf-8")
    assert "-H @-" in text, "API probe must pass the Authorization header via stdin (curl -H @-)"
    assert '-H "Authorization' not in text and "-H 'Authorization" not in text, (
        "Authorization header must never be a curl argv literal (pod-ps leak class)"
    )


# ---------------------------------------------------------------------------
# 6. VM-side fail-fast: no GITHUB_TOKEN in the local .env -> no SSH at all
# ---------------------------------------------------------------------------


def test_refresh_missing_local_github_token_fails_fast(harness: RefreshHarness) -> None:
    harness.env_file.write_text("HF_TOKEN=unrelated\n", encoding="utf-8")
    res = _run_refresh(harness.env)
    assert res.returncode != 0
    assert "no GITHUB_TOKEN" in res.stdout
    assert harness.log.read_text(encoding="utf-8") == "", (
        "must fail fast VM-side before any ssh/scp call"
    )


# ---------------------------------------------------------------------------
# 7. Static pin: the probe stays --dry-run (a dropped flag would create
#    junk branches on the remote)
# ---------------------------------------------------------------------------


def test_probe_dry_run_flag_pinned() -> None:
    text = SYNC_ENV_KEYS.read_text(encoding="utf-8")
    assert "push --dry-run origin HEAD:refs/heads/eps-auth-probe" in text


# ---------------------------------------------------------------------------
# 8. Single definition of the credential helper (the #1401 extraction)
# ---------------------------------------------------------------------------


def test_single_definition_of_helper() -> None:
    assign_files: list[str] = []
    for path in sorted(SCRIPTS.iterdir()):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(line.startswith('GIT_CRED_HELPER="') for line in text.splitlines()):
            assign_files.append(path.name)
    assert assign_files == ["_git_cred_helper.sh"], (
        f"exactly ONE GIT_CRED_HELPER assignment under scripts/ (the shared lib), "
        f"found: {assign_files}"
    )
    for consumer in (BOOTSTRAP, SYNC_ENV_KEYS):
        text = consumer.read_text(encoding="utf-8")
        assert 'source "$SCRIPT_DIR/_git_cred_helper.sh"' in text, (
            f"{consumer.name} must source the shared lib"
        )


# ---------------------------------------------------------------------------
# 9. Static pin: `sync env` failure path redirects to the refresh leg
# ---------------------------------------------------------------------------


def test_sync_env_403_hint_present() -> None:
    text = SYNC_ENV.read_text(encoding="utf-8")
    assert "keys --refresh-token" in text, (
        "sync_env.sh must hint at the git-auth repair leg on a 40x failure"
    )
