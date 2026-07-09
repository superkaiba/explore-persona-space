"""End-to-end tests for the ``.claude/hooks/guard_harmful_bank_read.sh`` PreToolUse hook.

The guard mechanizes CLAUDE.md § "Spurious usage-policy refusals" clause (d):
the six harmful safety-benchmark question banks under
``src/explore_persona_space/artifacts/query_banks/`` (advbench / strongreject /
betley_main8 / wang44 / broad_em_train / sensitive_info_requests) are
DIGEST-ONLY — a tool call that would page bank ITEM TEXT into the conversation
context is blocked (exit 2 + a ``BLOCKED`` stderr naming the allowed digest
alternatives), while digest ops (``jq 'length'`` / ``jq 'keys'`` / ``wc -l`` /
``sha256sum`` / ``grep -c``), in-process pipeline consumption
(``uv run python scripts/... <bank>``), and everything touching the benign
banks stay allowed (exit 0). Fail-OPEN on malformed input; escape hatch
``EPM_ALLOW_BANK_READ=1`` (session env or inline Bash prefix); every deny
appends one best-effort line to a sidecar log (``EPM_BANK_GUARD_LOG``) that
never affects the verdict.

These tests drive the script exactly as the harness does: stdin PreToolUse
JSON ``{"tool_name": ..., "tool_input": {...}}`` -> exit 2 (deny) or exit 0
(allow) — the subprocess-drives-script convention of
``tests/test_guard_repo_root_branch.py``. Env hygiene: deny cases run with
``EPM_ALLOW_BANK_READ`` scrubbed and the sidecar log pointed at ``/dev/null``;
the escape-hatch cases set the var explicitly; the log tests point
``EPM_BANK_GUARD_LOG`` at ``tmp_path``.

CONTENT HYGIENE (the very rule this hook mechanizes): no test reads or embeds
bank ITEM TEXT — deny cases only need paths, and allow cases feed
digest-shaped command STRINGS through the guard without executing them.

``TestSettingsWiring`` (plan #965 §6 case 20 / acceptance 2b) additionally
parses ``.claude/settings.json``, asserts the matcher group + command path +
executable bit, and invokes THE CONFIGURED COMMAND end-to-end — closing the
"hook ships green but inert" false-PASS channel (matcher typo / wrong command
path / missing +x). When this suite runs from a pre-merge worktree, the
canonical-root PREFIX of the configured absolute path is remapped onto this
checkout (the rest of the path is exercised verbatim, so a wrong directory /
basename / missing +x still fails); on the main checkout no remap occurs.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = _REPO_ROOT / ".claude" / "hooks" / "guard_harmful_bank_read.sh"
SETTINGS = _REPO_ROOT / ".claude" / "settings.json"
QUERY_BANKS = _REPO_ROOT / "src" / "explore_persona_space" / "artifacts" / "query_banks"
QUERY_BANKS_REL = "src/explore_persona_space/artifacts/query_banks"

HARMFUL_STEMS = [
    "advbench",
    "strongreject",
    "betley_main8",
    "wang44",
    "broad_em_train",
    "sensitive_info_requests",
]
BENIGN_FILES = [
    "arc_c_v1.json",
    "fact_questions_v1.json",
    "marker_eval_v1.json",
    "sycophancy_claims_v1.json",
    "wildchat_random_v1.json",
    "china_sensitive_v1.json",
    "README.md",
]

BANK_ABS = str(QUERY_BANKS / "advbench_v1.json")
BANK_REL = f"{QUERY_BANKS_REL}/advbench_v1.json"
MATCHER_TOOLS = {"Read", "Bash", "Grep", "mcp__ssh__ssh_execute"}


def _env(
    extra: dict[str, str] | None = None, *, allow: bool = False, log: str = "/dev/null"
) -> dict[str, str]:
    """Hook env: EPM_ALLOW_BANK_READ scrubbed (deny hygiene), sidecar log to /dev/null.

    ``allow=True`` sets the session-env escape hatch; ``log`` overrides the
    sidecar-log target (the log tests point it at tmp_path).
    """
    env = {k: v for k, v in os.environ.items() if k != "EPM_ALLOW_BANK_READ"}
    env["EPM_BANK_GUARD_LOG"] = log
    if allow:
        env["EPM_ALLOW_BANK_READ"] = "1"
    if extra:
        env.update(extra)
    return env


def _run(
    payload: dict | str,
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    script: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Feed a PreToolUse payload (dict -> JSON, str -> raw) to the guard."""
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run(
        [str(script or SCRIPT)],
        input=raw,
        text=True,
        capture_output=True,
        env=env if env is not None else _env(),
        cwd=cwd,
    )


def _run_read(file_path: str, **kw) -> subprocess.CompletedProcess[str]:
    return _run({"tool_name": "Read", "tool_input": {"file_path": file_path}}, **kw)


def _run_bash(cmd: str, **kw) -> subprocess.CompletedProcess[str]:
    return _run({"tool_name": "Bash", "tool_input": {"command": cmd}}, **kw)


def _run_grep(path: str, output_mode: str = "content", **kw) -> subprocess.CompletedProcess[str]:
    payload = {"tool_name": "Grep", "tool_input": {"pattern": "foo", "path": path}}
    if output_mode is not None:
        payload["tool_input"]["output_mode"] = output_mode
    return _run(payload, **kw)


def _assert_denied(r: subprocess.CompletedProcess[str]) -> None:
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "BLOCKED" in r.stderr, r.stderr


def _assert_allowed(r: subprocess.CompletedProcess[str]) -> None:
    assert r.returncode == 0, (r.returncode, r.stderr)


# ---------------------------------------------------------------------------
# DENY — Read arm (§6 cases 1-4, 10j)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stem", HARMFUL_STEMS)
def test_read_denied_each_bank_absolute(stem: str) -> None:
    _assert_denied(_run_read(str(QUERY_BANKS / f"{stem}_v1.json")))


def test_read_denied_repo_relative_path() -> None:
    _assert_denied(_run_read(BANK_REL))


def test_read_denied_version_bumped_nonexistent() -> None:
    # Raw-regex arm: the file does not exist; the name shape alone denies.
    _assert_denied(_run_read(str(QUERY_BANKS / "advbench_v2.json")))


def test_read_denied_worktree_copy_path() -> None:
    p = (
        "/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-9/"
        f"{QUERY_BANKS_REL}/strongreject_v1.json"
    )
    _assert_denied(_run_read(p))


def test_tool_name_absent_bank_file_path_denied() -> None:
    # §6 case 10j: Read-arm fallback dispatch on field shape when tool_name is absent.
    _assert_denied(_run({"tool_input": {"file_path": BANK_ABS}}))


def test_deny_message_does_not_print_override() -> None:
    # Plan §11.13: the deny message must NOT contain the literal override
    # incantation (it is documented in CLAUDE.md clause (d) instead).
    r = _run_read(BANK_ABS)
    _assert_denied(r)
    assert "EPM_ALLOW_BANK_READ" not in r.stderr, r.stderr


# ---------------------------------------------------------------------------
# DENY — Bash arm: text-paging verbs (§6 cases 5-8, 10, 10h)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        f"cat {BANK_ABS}",
        f"head -5 {BANK_ABS}",
        f"tail -n 3 {BANK_ABS}",
        f"sed -n '1,10p' {BANK_ABS}",
        "awk '{print}' " + BANK_ABS,
    ],
)
def test_bash_paging_verbs_denied(cmd: str) -> None:
    _assert_denied(_run_bash(cmd))


def test_bash_pipe_does_not_exempt() -> None:
    _assert_denied(_run_bash(f"cat {BANK_ABS} | grep foo"))


@pytest.mark.parametrize(
    "cmd",
    [
        f"jq '.' {BANK_ABS}",
        f"jq -r '.[]' {BANK_ABS}",
        f"jq '.[0]' {BANK_ABS}",
    ],
)
def test_bash_jq_item_access_denied(cmd: str) -> None:
    _assert_denied(_run_bash(cmd))


def test_bash_grep_line_output_denied() -> None:
    _assert_denied(_run_bash(f"grep harmful {BANK_ABS}"))


def test_bash_pod_style_absolute_path_denied() -> None:
    _assert_denied(
        _run_bash(f"cat /workspace/explore-persona-space/{QUERY_BANKS_REL}/wang44_v1.json")
    )


def test_bash_python_json_tool_denied() -> None:
    _assert_denied(_run_bash(f"uv run python -m json.tool {BANK_ABS}"))


# ---------------------------------------------------------------------------
# DENY — realpath branch (§6 cases 10b, 10c)
# ---------------------------------------------------------------------------
def test_bash_cwd_relative_bank_denied_via_realpath() -> None:
    # §6 case 10b: bare basename carries no query_banks/ component — only the
    # realpath resolution (against the subprocess cwd) can catch it.
    _assert_denied(_run_bash("cat advbench_v1.json", cwd=str(QUERY_BANKS)))


def test_read_symlink_to_bank_denied(tmp_path: Path) -> None:
    link = tmp_path / "advbench_link.json"
    link.symlink_to(QUERY_BANKS / "advbench_v1.json")
    _assert_denied(_run_read(str(link)))


def test_bash_cat_symlink_to_bank_denied(tmp_path: Path) -> None:
    link = tmp_path / "advbench_link.json"
    link.symlink_to(QUERY_BANKS / "advbench_v1.json")
    _assert_denied(_run_bash(f"cat {link}"))


# ---------------------------------------------------------------------------
# DENY — operator normalization + per-instance tracking (§6 cases 10d-10g)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        f"cat {BANK_ABS}|grep foo",
        f"cat {BANK_ABS}&&echo done",
        f"cat {BANK_ABS};echo done",
    ],
)
def test_bash_unspaced_operators_denied(cmd: str) -> None:
    _assert_denied(_run_bash(cmd))


def test_bash_rg_dash_L_is_follow_not_digest() -> None:
    # ripgrep -L is --follow (prints matching lines), NOT files-without-match.
    _assert_denied(_run_bash(f"rg -L foo {BANK_ABS}"))


def test_bash_rg_line_output_denied() -> None:
    _assert_denied(_run_bash(f"rg foo {BANK_ABS}"))


def test_bash_per_instance_grep_no_laundering() -> None:
    # §6 case 10f: the second instance's -c must not launder the first.
    _assert_denied(_run_bash(f"grep harmful {BANK_ABS} && grep -c foo /tmp/other.txt"))


def test_bash_per_instance_jq_no_laundering() -> None:
    # A later digest jq must not launder an earlier item-access jq.
    _assert_denied(_run_bash(f"jq '.' {BANK_ABS} && jq 'length' {BANK_ABS}"))


@pytest.mark.parametrize(
    "cmd",
    [
        f"git diff -- {BANK_ABS}",
        f"git show HEAD:{QUERY_BANKS_REL}/advbench_v1.json",
        f"git log -p -- {BANK_ABS}",
    ],
)
def test_bash_git_paging_denied(cmd: str) -> None:
    _assert_denied(_run_bash(cmd))


def test_bash_per_instance_git_no_laundering() -> None:
    # An earlier digest git must not launder a later git-show paging instance.
    _assert_denied(
        _run_bash(
            f"git log --oneline -- {BANK_ABS} && git show HEAD:{QUERY_BANKS_REL}/advbench_v1.json"
        )
    )


# ---------------------------------------------------------------------------
# DENY — Grep tool + ssh_execute arms (§6 cases 9, 10i, 10k)
# ---------------------------------------------------------------------------
def test_grep_tool_content_mode_on_bank_denied() -> None:
    _assert_denied(_run_grep(BANK_ABS, output_mode="content"))


def test_grep_tool_content_mode_on_bank_dir_denied() -> None:
    _assert_denied(_run_grep(str(QUERY_BANKS), output_mode="content"))


def test_ssh_execute_arm_denied() -> None:
    # §6 case 10i: pod-side cat pages text into context identically.
    payload = {
        "tool_name": "mcp__ssh__ssh_execute",
        "tool_input": {
            "command": f"cat /workspace/explore-persona-space/{QUERY_BANKS_REL}/advbench_v1.json"
        },
    }
    _assert_denied(_run(payload))


# ---------------------------------------------------------------------------
# ALLOW — benign banks + aggregate files (§6 cases 11, 12)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", BENIGN_FILES)
def test_read_benign_bank_allowed(name: str) -> None:
    _assert_allowed(_run_read(str(QUERY_BANKS / name)))


@pytest.mark.parametrize("name", BENIGN_FILES)
def test_bash_cat_benign_bank_allowed(name: str) -> None:
    _assert_allowed(_run_bash(f"cat {QUERY_BANKS / name}"))


def test_read_aggregate_eval_result_allowed() -> None:
    # §6 case 12: eval_results aggregates sharing a bank stem are BLESSED —
    # no query_banks/ dir component, so the raw regex + realpath both miss.
    _assert_allowed(_run_read(str(_REPO_ROOT / "eval_results/issue_123/advbench_summary.json")))


# ---------------------------------------------------------------------------
# ALLOW — Bash digests + pipeline consumption + unrelated (§6 cases 13-15)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        f"jq 'length' {BANK_ABS}",
        f"jq 'keys' {BANK_ABS}",
        f"jq keys {BANK_ABS}",
        f"jq 'type' {BANK_ABS}",
        f"wc -l {BANK_ABS}",
        f"sha256sum {BANK_ABS}",
        f"md5sum {BANK_ABS}",
        f"ls -la {BANK_ABS}",
        f"stat {BANK_ABS}",
        f"du -h {BANK_ABS}",
        f"file {BANK_ABS}",
        f"grep -c foo {BANK_ABS}",
        f"grep -q foo {BANK_ABS}",
        f"grep -l foo {BANK_ABS}",
        f"grep -L foo {BANK_ABS}",  # GNU grep files-without-match IS a digest
        f"rg -c foo {BANK_ABS}",
        f"rg --files-without-match foo {BANK_ABS}",
        f"git diff --stat -- {BANK_ABS}",
        f"git log --oneline -- {BANK_ABS}",
    ],
)
def test_bash_digest_ops_allowed(cmd: str) -> None:
    _assert_allowed(_run_bash(cmd))


@pytest.mark.parametrize(
    "cmd",
    [
        f"uv run python scripts/eval.py --bank {BANK_REL}",
        f"git add {BANK_REL}",
        f"cp {BANK_ABS} /tmp/x",
    ],
)
def test_bash_pipeline_consumption_allowed(cmd: str) -> None:
    _assert_allowed(_run_bash(cmd))


@pytest.mark.parametrize("cmd", ["cat notes.md", "git status", ""])
def test_bash_unrelated_commands_allowed(cmd: str) -> None:
    _assert_allowed(_run_bash(cmd))


# ---------------------------------------------------------------------------
# ALLOW — Grep tool digest modes (§6 case 16)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["files_with_matches", "count"])
@pytest.mark.parametrize("path", [BANK_ABS, str(QUERY_BANKS)])
def test_grep_tool_digest_modes_allowed(path: str, mode: str) -> None:
    _assert_allowed(_run_grep(path, output_mode=mode))


def test_grep_tool_default_mode_allowed() -> None:
    # output_mode absent defaults to files_with_matches -> digest -> allowed.
    payload = {"tool_name": "Grep", "tool_input": {"pattern": "foo", "path": BANK_ABS}}
    _assert_allowed(_run(payload))


def test_grep_tool_content_mode_on_non_bank_allowed() -> None:
    _assert_allowed(_run_grep(str(QUERY_BANKS / "arc_c_v1.json"), output_mode="content"))


# ---------------------------------------------------------------------------
# Escape hatch (§6 case 17)
# ---------------------------------------------------------------------------
def test_inline_escape_hatch_allows_bash() -> None:
    _assert_allowed(_run_bash(f"EPM_ALLOW_BANK_READ=1 cat {BANK_ABS}"))


def test_session_env_escape_hatch_allows_bash() -> None:
    _assert_allowed(_run_bash(f"cat {BANK_ABS}", env=_env(allow=True)))


def test_session_env_escape_hatch_allows_read() -> None:
    _assert_allowed(_run_read(BANK_ABS, env=_env(allow=True)))


# ---------------------------------------------------------------------------
# Fail-open (§6 case 18)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw",
    [
        "this is not json {",
        "",
        json.dumps({"tool_name": "Bash"}),  # no tool_input
        json.dumps({"tool_name": "SomeOtherTool", "tool_input": {"irrelevant": 1}}),
    ],
)
def test_fail_open_on_malformed_or_irrelevant_input(raw: str) -> None:
    _assert_allowed(_run(raw))


# ---------------------------------------------------------------------------
# Deny sidecar log (§6 case 19)
# ---------------------------------------------------------------------------
def test_deny_appends_exactly_one_log_line(tmp_path: Path) -> None:
    log = tmp_path / "denies.log"
    r = _run_read(BANK_ABS, env=_env(log=str(log)))
    _assert_denied(r)
    lines = log.read_text().splitlines()
    assert len(lines) == 1, lines
    assert len(lines[0].split("\t")) == 3, lines[0]  # ts \t what \t target


def test_unwritable_log_path_still_denies(tmp_path: Path) -> None:
    # Log failure must never change the verdict (best-effort || true).
    log = tmp_path / "no_such_dir" / "denies.log"
    _assert_denied(_run_read(BANK_ABS, env=_env(log=str(log))))


def test_allow_writes_no_log_line(tmp_path: Path) -> None:
    log = tmp_path / "denies.log"
    _assert_allowed(_run_bash(f"jq 'length' {BANK_ABS}", env=_env(log=str(log))))
    assert not log.exists()


# ---------------------------------------------------------------------------
# Settings WIRING (§6 case 20 / acceptance 2b — the inert-hook false-PASS closer)
# ---------------------------------------------------------------------------
def _main_repo_root() -> str | None:
    """Canonical main-checkout root (parent of the shared .git common dir)."""
    r = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return str(Path(r.stdout.strip()).parent)


class TestSettingsWiring:
    """Parse .claude/settings.json and invoke the CONFIGURED command.

    Every other test drives ``SCRIPT`` directly; without this class a matcher
    typo / wrong command path / missing +x ships the mechanical layer inert
    with a green suite. The command path is read FROM settings, never from
    the repo constant. Pre-merge worktree runs remap ONLY the canonical-root
    prefix onto this checkout (see module docstring); wrong-directory /
    wrong-basename / missing-+x bugs still fail under the remap.
    """

    def _bank_guard_entry(self) -> dict:
        settings = json.loads(SETTINGS.read_text())
        for entry in settings["hooks"]["PreToolUse"]:
            matcher = entry.get("matcher", "")
            if set(matcher.split("|")) == MATCHER_TOOLS:
                return entry
        pytest.fail(
            "no hooks.PreToolUse entry whose matcher alternation covers exactly "
            f"{sorted(MATCHER_TOOLS)}"
        )

    def _configured_command(self) -> Path:
        entry = self._bank_guard_entry()
        cmds = [h["command"] for h in entry.get("hooks", []) if h.get("type") == "command"]
        assert len(cmds) == 1, cmds
        cmd = cmds[0]
        assert os.path.isabs(cmd), cmd
        assert os.path.basename(cmd) == "guard_harmful_bank_read.sh", cmd
        main_root = _main_repo_root()
        if main_root is not None and str(_REPO_ROOT) != main_root:
            prefix = main_root.rstrip("/") + "/"
            if cmd.startswith(prefix):
                cmd = str(_REPO_ROOT / cmd[len(prefix) :])
        return Path(cmd)

    def test_matcher_group_present_with_single_command_hook(self) -> None:
        self._configured_command()

    def test_configured_command_exists_and_is_executable(self) -> None:
        cmd = self._configured_command()
        assert cmd.exists(), cmd
        assert os.access(cmd, os.X_OK), cmd  # mechanizes acceptance 1's chmod +x

    def test_configured_command_denies_bank_read(self) -> None:
        r = _run_read(BANK_ABS, script=self._configured_command())
        _assert_denied(r)

    def test_configured_command_allows_digest_bash(self) -> None:
        r = _run_bash(f"jq 'length' {BANK_ABS}", script=self._configured_command())
        _assert_allowed(r)


# ---------------------------------------------------------------------------
# #1152 (a) DENY — cross-unit flag laundering closed (plan #1152 §6 cases 1-8)
#
# Operator tokens `| & ; ( )` + backtick (newlines tr'd to `;`) are INSTANCE
# BOUNDARIES: per-instance grep/git/jq safe-flag state closes there, so a
# later command unit's flags can no longer mark an earlier unit's instance
# safe. The DENY co-occurrence (bank + verb) stays whole-command (#965 §11.3).
# ---------------------------------------------------------------------------
TASKPY = "uv run python scripts/task.py"


@pytest.mark.parametrize(
    "cmd",
    [
        f"grep harmful {BANK_ABS} && ls -l",  # 1: `ls -l` laundered the grep pre-#1152
        f"grep harmful {BANK_ABS}; ls -l",  # 2: `;` separator
        f"grep harmful {BANK_ABS} | wc -l",  # 3: `wc -l` laundered through the pipe
        f"rg foo {BANK_ABS} || echo -l",  # 4: `||` separator, rg family
        f"git show HEAD:{BANK_REL} && ls --stat",  # 5: later --stat laundered git show
        f"grep harmful {BANK_ABS}\nls -l",  # 6: real newline boundary (tr maps to `;`)
        # 7: pins that the DENY side stayed WHOLE-COMMAND (#965 §11.3 invariant —
        # this deliberate-FP shape must remain a deny after the boundary change).
        f"jq 'length' {BANK_ABS} && jq '.foo' /tmp/other.json",
        # 8: pins the accepted new FP as intentional (plan #1152 §8): a digest flag
        # AFTER a quoted alternation — the quoted `|` pads into a boundary that
        # latches the still-unsafe instance. Remediation: put -c before the pattern.
        f"grep -E 'foo|bar' -c {BANK_ABS}",
    ],
)
def test_bash_boundary_laundering_denied(cmd: str) -> None:
    _assert_denied(_run_bash(cmd))


# ---------------------------------------------------------------------------
# #1152 (a) ALLOW — no new FPs on documented digests (§6 cases 9-14)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        f"grep -c foo {BANK_ABS} && ls -l",  # 9: safe latched before the boundary
        f"grep -q foo {BANK_ABS} && grep -c bar {BANK_ABS}",  # 10: two safe instances
        f"grep -cE 'foo|bar' {BANK_ABS}",  # 11: flag latched before the quoted `|`
        f"git diff --stat -- {BANK_ABS} && git log --oneline -- {BANK_ABS}",  # 12
        f"git log -- {BANK_ABS} && mkdir -p /tmp/x",  # 13: cross-command -p FALSE-DENY fixed
        f"wc -l {BANK_ABS} && echo done",  # 14
    ],
)
def test_bash_boundary_no_new_false_positives(cmd: str) -> None:
    _assert_allowed(_run_bash(cmd))


# ---------------------------------------------------------------------------
# #1152 (b) — bare diff / comm / join page bank content (§6 cases 15-19, 21;
# case 20 `git diff --stat -- {B}` is already parametrized in
# test_bash_digest_ops_allowed above and stays untouched)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        f"diff /dev/null {BANK_ABS}",  # 15: prints every item
        f"diff {BANK_ABS} /tmp/other.json",  # 16
        f"comm {BANK_ABS} /dev/null",  # 17
        f"join {BANK_ABS} /tmp/x",  # 18
        # 19: (a)+(b) compose — the resolved git instance closes at the
        # boundary, so the second unit's diff is bare (in_git=0).
        f"git log --oneline -- {BANK_ABS} && diff /dev/null {BANK_ABS}",
    ],
)
def test_bash_bare_diff_comm_join_denied(cmd: str) -> None:
    _assert_denied(_run_bash(cmd))


def test_bash_git_prefixed_diff_digest_allowed() -> None:
    # 21: the in_git guard keeps git-prefixed diff digests out of the
    # bare-diff deny (the same-unit `git` token precedes its `diff` token).
    _assert_allowed(_run_bash(f"git -C /tmp diff --numstat -- {BANK_ABS}"))


# ---------------------------------------------------------------------------
# #1152 (a)-v2 — unresolved-git-instance CARRY across boundaries (§6 35-38)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        # 35: the padded `(` of $(pwd) fires a boundary BETWEEN `git` and its
        # subcommand — a naive reset would detach the attribution (fail-OPEN).
        f"git -C $(pwd) show HEAD:{BANK_REL}",
        # 36: backslash-continuation newline -> `;` boundary mid-instance;
        # bash executes this as ONE `git show`.
        f"git \\\n show HEAD:{BANK_REL}",
        # 37
        f"git -C $(pwd) log -p -- {BANK_ABS}",
    ],
)
def test_bash_git_carry_denies_paging(cmd: str) -> None:
    _assert_denied(_run_bash(cmd))


def test_bash_git_carry_preserves_diff_stat_digest() -> None:
    # 38: the carry also restores the digest allow a naive boundary reset
    # would false-deny via the bare-diff branch.
    _assert_allowed(_run_bash(f"git -C $(pwd) diff --stat -- {BANK_ABS}"))


# ---------------------------------------------------------------------------
# #1152 boundary robustness (§6 cases 39-44) + backtick-substitution padding
# ---------------------------------------------------------------------------
def test_bash_pipe_amp_boundary_denied() -> None:
    # 39: `|&` pads to two consecutive boundary tokens.
    _assert_denied(_run_bash(f"grep harmful {BANK_ABS} |& wc -l"))


def test_bash_unsafe_grep_in_last_unit_denied() -> None:
    # 41: reverse direction — the unsafe grep is the LAST unit, closed by the
    # pre-existing end-of-walk latch rather than a boundary.
    _assert_denied(_run_bash(f"ls -l && grep harmful {BANK_ABS}"))


@pytest.mark.parametrize(
    "cmd",
    [
        f"jq 'length' {BANK_ABS} && echo done",  # 40: digest jq closes clean
        f"grep -c foo {BANK_ABS} 2>&1",  # 42: redirect tokens after a latched safe flag
        f"if grep -q foo {BANK_ABS}; then echo hit; fi",  # 43: shell-keyword units
        f"n=$(grep -c foo {BANK_ABS})",  # 44: digest inside command substitution
        f"n=`grep -c foo {BANK_ABS}`",  # backtick substitution of a digest grep
    ],
)
def test_bash_boundary_shapes_allowed(cmd: str) -> None:
    _assert_allowed(_run_bash(cmd))


def test_bash_backtick_substitution_paging_denied() -> None:
    # Backticks pad + bound like `$( )` parens: pre-#1152 the backtick glued
    # to the verb and to the closing path token, so the walk saw neither the
    # verb nor a bank token and failed OPEN on a bare backtick-cat of a bank.
    _assert_denied(_run_bash(f"echo `cat {BANK_ABS}`"))


# ---------------------------------------------------------------------------
# #1152 (c) ALLOW — task.py --note quoted-prose exemption (§6 cases 22-26).
# Notes are DATA arguments to a Python CLI, never executed: quoted --note
# strings on a task.py-shaped command are blanked before the fast path.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        # 22: double-quoted descriptive note naming a paging verb + a bank path
        f'{TASKPY} post-marker 999 epm:progress --note "ran sed over {BANK_REL} digest"',
        # 23: single-quoted note
        f"{TASKPY} post-marker 999 epm:progress --note 'grep pass over {BANK_REL}: 0 hits'",
        # 24: set-status --note (identical flag on the identical CLI)
        f'{TASKPY} set-status 42 blocked --note "cat of {BANK_REL} denied by guard"',
        # 25: --note= glued form
        f'{TASKPY} post-marker 1 epm:x --note="sed over {BANK_REL}"',
        # 26: the ONLY bank token lives in the note -> scrubbed fast path allows
        f'sed -i s/a/b/ /tmp/f.txt && {TASKPY} post-marker 1 epm:x --note "checked {BANK_REL}"',
    ],
)
def test_bash_taskpy_note_prose_allowed(cmd: str) -> None:
    _assert_allowed(_run_bash(cmd))


# ---------------------------------------------------------------------------
# #1152 (c) DENY — the note exemption is not launderable (§6 cases 27-30)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        # 27: verb + bank OUTSIDE the note survive the scrub
        f'cat {BANK_ABS} && {TASKPY} post-marker 1 epm:x --note "done"',
        # 28: $(...) inside a double-quoted note EXECUTES -> never blanked
        f'{TASKPY} post-marker 1 epm:x --note "summary: $(cat {BANK_ABS})"',
        # 29: backtick substitution inside a note EXECUTES -> never blanked
        f'{TASKPY} post-marker 1 epm:x --note "summary: `cat {BANK_ABS}`"',
        # 30: no task.py in the command -> the scrub gate never fires (pins
        # the GATE, not just the sed)
        f'echo --note "{BANK_REL}" | xargs -n1 cat',
    ],
)
def test_bash_taskpy_note_exemption_not_launderable(cmd: str) -> None:
    _assert_denied(_run_bash(cmd))


# ---------------------------------------------------------------------------
# #1152 rider — task.py --file/--body-file on a bank denies (§6 cases 31-34):
# it would embed bank items into task state (events.jsonl / body.md / plans).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        f"{TASKPY} post-marker 1 epm:x --file {BANK_ABS}",  # 31
        f"{TASKPY} post-marker 1 epm:x --file={BANK_ABS}",  # 32: glued form
    ],
)
def test_bash_taskpy_file_on_bank_denied(cmd: str) -> None:
    _assert_denied(_run_bash(cmd))


@pytest.mark.parametrize(
    "cmd",
    [
        f"{TASKPY} post-marker 1 epm:x --file /tmp/note.md",  # 33: non-bank file
        f"uv run python scripts/eval.py --file {BANK_ABS}",  # 34: non-task.py untouched
    ],
)
def test_bash_taskpy_file_non_bank_or_non_taskpy_allowed(cmd: str) -> None:
    _assert_allowed(_run_bash(cmd))
