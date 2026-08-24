"""Tests for scripts/codex_auto_upgrade.py + its cron wrapper (task #2322).

Covers the pure decision helpers, plus four end-to-end pins added at
landing time: main() under --dry-run (mutation seams intercepted), the
wrapper's fail-loud cd branch (rewritten copy in tmp_path), the
broken-current-model exclude-and-replace ordering, and the wrapper's
alert-on-setup-failure path (missing prerequisite + unwritable log dir,
the two sites that used to `exit` before the alert arm). No test issues a real
`codex exec` probe, touches the live ~/.codex/config.toml, installs
anything into the npm global prefix, kills any live process, or pushes to
the real Telegram channel.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import create_autospec

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    path = REPO_ROOT / "scripts" / "codex_auto_upgrade.py"
    spec = importlib.util.spec_from_file_location("codex_auto_upgrade", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


# --------------------------------------------------------------------------
# parse_version
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("codex-cli 0.147.0", (0, 147, 0)),
        ("0.147.0\n", (0, 147, 0)),
        ("0.144.0", (0, 144, 0)),
        ("1.2", (1, 2)),
    ],
)
def test_parse_version_extracts_dotted_numeric(text, expected):
    assert mod.parse_version(text) == expected


@pytest.mark.parametrize("text", ["", "no digits here", None])
def test_parse_version_unparseable_is_none_not_zero(text):
    """None, never (0,), so an unreadable version cannot masquerade as
    'ancient' and trigger a spurious upgrade."""
    assert mod.parse_version(text) is None


def test_version_ordering_is_numeric_not_lexicographic():
    # The bug this guards: "0.9.0" > "0.147.0" under string comparison.
    assert mod.parse_version("0.147.0") > mod.parse_version("0.9.0")
    assert mod.parse_version("0.147.0") > mod.parse_version("0.144.0")


# --------------------------------------------------------------------------
# config read/write
# --------------------------------------------------------------------------


CONFIG = """model = "gpt-5.5"
model_reasoning_effort = "high"
service_tier = "default"

[features]
goals = true

[projects."/home/x/repo"]
trust_level = "trusted"
"""


def test_read_config_model_reads_top_level_key():
    assert mod.read_config_model(CONFIG) == "gpt-5.5"


def test_write_config_model_replaces_only_the_model_line():
    out = mod.write_config_model(CONFIG, "gpt-5.6-sol")
    assert mod.read_config_model(out) == "gpt-5.6-sol"
    # Everything else byte-preserved — notably the effort, which is a user
    # cost preference the upgrader must never touch.
    assert 'model_reasoning_effort = "high"' in out
    assert 'trust_level = "trusted"' in out
    assert "[features]" in out
    assert len(out.splitlines()) == len(CONFIG.splitlines())


def test_write_config_model_ignores_model_keys_inside_sections():
    """A [profiles.*] model override must survive: only the top-level key is
    the twin's model."""
    cfg = 'model = "a"\n\n[profiles.deep]\nmodel = "b"\n'
    out = mod.write_config_model(cfg, "c")
    assert out == 'model = "c"\n\n[profiles.deep]\nmodel = "b"\n'


def test_read_config_model_absent_top_level_is_none():
    assert mod.read_config_model('[profiles.deep]\nmodel = "b"\n') is None


def test_write_config_model_prepends_when_absent():
    out = mod.write_config_model("[features]\ngoals = true\n", "gpt-5.6-sol")
    assert out.startswith('model = "gpt-5.6-sol"\n')
    assert mod.read_config_model(out) == "gpt-5.6-sol"


def test_atomic_write_leaves_no_orphan_tmp(tmp_path):
    target = tmp_path / "config.toml"
    target.write_text("old")
    mod.atomic_write(target, "new")
    assert target.read_text() == "new"
    assert list(tmp_path.iterdir()) == [target]


# --------------------------------------------------------------------------
# candidate_models
# --------------------------------------------------------------------------


def _m(slug, priority, minver="0.98.0", visibility="list"):
    return {
        "slug": slug,
        "priority": priority,
        "minimal_client_version": minver,
        "visibility": visibility,
    }


def test_candidates_sorted_by_priority_best_first():
    models = [_m("mid", 7), _m("best", 1), _m("worst", 16)]
    got = [m["slug"] for m in mod.candidate_models(models, (0, 147, 0), {})]
    assert got == ["best", "mid", "worst"]


def test_candidates_exclude_models_needing_a_newer_cli():
    """The gpt-5.6-sol case: listed, but a 400 at dispatch on an old CLI."""
    models = [_m("too-new", 1, minver="0.144.0"), _m("ok", 7, minver="0.98.0")]
    got = [m["slug"] for m in mod.candidate_models(models, (0, 137, 0), {})]
    assert got == ["ok"]
    # ...and become eligible once the CLI is new enough.
    got2 = [m["slug"] for m in mod.candidate_models(models, (0, 147, 0), {})]
    assert got2 == ["too-new", "ok"]


def test_candidates_exclude_hidden_models():
    models = [_m("internal", 1, visibility="hide"), _m("public", 7)]
    got = [m["slug"] for m in mod.candidate_models(models, (0, 147, 0), {})]
    assert got == ["public"]


def test_candidates_exclude_known_bad_for_same_cli_version():
    """A slug that 400'd on this CLI is skipped without re-probing."""
    models = [_m("bad", 1), _m("good", 7)]
    known_bad = {"bad": {"cli_version": "0.147.0", "error": "not supported"}}
    got = [m["slug"] for m in mod.candidate_models(models, (0, 147, 0), known_bad)]
    assert got == ["good"]


def test_known_bad_is_rechecked_after_a_cli_upgrade():
    """Keyed by CLI version so a newer CLI re-probes a previously bad slug —
    otherwise one bad day would blacklist a model permanently."""
    models = [_m("bad", 1), _m("good", 7)]
    known_bad = {"bad": {"cli_version": "0.140.0", "error": "not supported"}}
    got = [m["slug"] for m in mod.candidate_models(models, (0, 147, 0), known_bad)]
    assert got == ["bad", "good"]


def test_candidates_tolerate_missing_priority_and_slug():
    models = [{"visibility": "list"}, _m("real", 3), {"slug": "np", "visibility": "list"}]
    got = [m["slug"] for m in mod.candidate_models(models, (0, 147, 0), {})]
    assert got == ["real", "np"]  # priority-less sorts last, slug-less dropped


# --------------------------------------------------------------------------
# inflight_jobs — the staleness bound
# --------------------------------------------------------------------------


def _write_state(tmp_path, name, jobs):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps({"jobs": jobs}))
    return d


def _iso(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) + ".000Z"


def test_fresh_nonterminal_job_counts_as_inflight(tmp_path, monkeypatch):
    _write_state(
        tmp_path, "ws", [{"id": "live", "phase": "running", "updatedAt": _iso(time.time() - 60)}]
    )
    monkeypatch.setattr(mod, "COMPANION_STATE_DIR", tmp_path)
    assert mod.inflight_jobs() == ["live"]


def test_stale_nonterminal_job_is_ignored(tmp_path, monkeypatch):
    """The observed debris: sessions killed mid-job strand records at
    'running' forever. Counting them would silently disable auto-upgrade."""
    _write_state(
        tmp_path,
        "ws",
        [{"id": "may-debris", "phase": "running", "updatedAt": "2026-05-15T05:43:45.977Z"}],
    )
    monkeypatch.setattr(mod, "COMPANION_STATE_DIR", tmp_path)
    assert mod.inflight_jobs() == []


def test_terminal_jobs_never_count(tmp_path, monkeypatch):
    now = time.time()
    _write_state(
        tmp_path,
        "ws",
        [
            {"id": "a", "phase": "done", "updatedAt": _iso(now)},
            {"id": "b", "phase": "failed", "updatedAt": _iso(now)},
            {"id": "c", "phase": "cancelled", "updatedAt": _iso(now)},
        ],
    )
    monkeypatch.setattr(mod, "COMPANION_STATE_DIR", tmp_path)
    assert mod.inflight_jobs() == []


def test_unknown_future_phase_counts_as_busy_when_fresh(tmp_path, monkeypatch):
    """An unrecognized phase must read as busy, not idle."""
    _write_state(
        tmp_path,
        "ws",
        [{"id": "x", "phase": "some-new-phase", "updatedAt": _iso(time.time() - 30)}],
    )
    monkeypatch.setattr(mod, "COMPANION_STATE_DIR", tmp_path)
    assert mod.inflight_jobs() == ["x"]


def test_epoch_ms_timestamps_are_understood(tmp_path, monkeypatch):
    _write_state(
        tmp_path, "ws", [{"id": "ms", "phase": "running", "updatedAt": (time.time() - 60) * 1000}]
    )
    monkeypatch.setattr(mod, "COMPANION_STATE_DIR", tmp_path)
    assert mod.inflight_jobs() == ["ms"]


def test_corrupt_state_file_does_not_wedge_upgrades_forever(tmp_path, monkeypatch):
    d = tmp_path / "ws"
    d.mkdir()
    sf = d / "state.json"
    sf.write_text("{not json")
    monkeypatch.setattr(mod, "COMPANION_STATE_DIR", tmp_path)
    # Fresh corruption is treated as busy (something may be mid-write)...
    assert len(mod.inflight_jobs()) == 1
    # ...but stale corruption is ignored, so it cannot disable the cron.
    old = time.time() - 10 * 3600
    import os as _os

    _os.utime(sf, (old, old))
    assert mod.inflight_jobs() == []


def test_missing_state_dir_is_not_busy(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "COMPANION_STATE_DIR", tmp_path / "nope")
    assert mod.inflight_jobs() == []


# --------------------------------------------------------------------------
# main() end-to-end + the cron wrapper (the four #2322 landing pins)
# --------------------------------------------------------------------------


def _fake_which(cmd, mode=os.F_OK | os.X_OK, path=None):
    """Signature-conformant shutil.which stand-in: every binary resolves."""
    return f"/stub/bin/{cmd}"


class _RunRecorder:
    """Signature-conformant fake for mod.run (mirrors run(cmd, timeout, cwd)).

    Routes on the command's leading tokens and records every invocation, so
    the tests can assert which side-effecting commands were — and were NOT —
    issued. Raises on a command it does not recognize (fail loud, never a
    silent default).
    """

    def __init__(
        self,
        cli_version: str = "codex-cli 0.147.0",
        npm_latest: str = "0.147.0",
        probe_results: dict[str, bool] | None = None,
        pgrep_pids: str = "",
    ) -> None:
        self.calls: list[list[str]] = []
        self.cli_version = cli_version
        self.npm_latest = npm_latest
        self.probe_results = probe_results or {}
        self.pgrep_pids = pgrep_pids

    def __call__(self, cmd, timeout=120, cwd=None):
        self.calls.append(list(cmd))
        if cmd[:2] == ["codex", "--version"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=self.cli_version, stderr="")
        if cmd[:2] == ["npm", "view"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=self.npm_latest, stderr="")
        if cmd[:2] == ["npm", "install"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["codex", "exec"]:
            slug = next((t[6:] for t in cmd if t.startswith("model=")), None)
            if self.probe_results.get(slug, False):
                out = f"{mod.PROBE_SENTINEL}\n{mod.PROBE_SENTINEL}\n"
                return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr='{"message": "probe declined by test fake"}'
            )
        if cmd[0] == "pgrep":
            rc = 0 if self.pgrep_pids else 1
            return subprocess.CompletedProcess(cmd, rc, stdout=self.pgrep_pids, stderr="")
        raise AssertionError(f"unexpected command routed through run(): {cmd}")

    def cmds(self, *prefix: str) -> list[list[str]]:
        p = list(prefix)
        return [c for c in self.calls if c[: len(p)] == p]

    def probed_slugs(self) -> list[str]:
        return [t[6:] for c in self.cmds("codex", "exec") for t in c if t.startswith("model=")]


def _wire_main(monkeypatch, tmp_path, recorder, models, config_text, argv):
    """Point every filesystem / network / process seam at tmp_path-local fakes.

    Returns (config path, the intercepted direct-subprocess.run kill seam).
    The kill seam is autospec'd from the real subprocess.run so a broken
    dry_run/guard thread is DETECTED (recorded call) rather than executed
    against live pids.
    """
    cfg = tmp_path / "config.toml"
    cfg.write_text(config_text)
    monkeypatch.setattr(mod, "run", recorder)
    monkeypatch.setattr(mod, "CONFIG_PATH", cfg)
    monkeypatch.setattr(mod, "KNOWN_BAD_PATH", tmp_path / "known-bad.json")
    monkeypatch.setattr(mod, "COMPANION_STATE_DIR", tmp_path / "no-companion-state")
    monkeypatch.setattr(mod.shutil, "which", _fake_which)

    def fake_fetch_models(client_version):  # mirrors fetch_models(client_version)
        return models

    monkeypatch.setattr(mod, "fetch_models", fake_fetch_models)
    kill_seam = create_autospec(
        subprocess.run, return_value=subprocess.CompletedProcess(["kill"], 0, "", "")
    )
    monkeypatch.setattr(mod.subprocess, "run", kill_seam)
    monkeypatch.setattr(sys, "argv", ["codex_auto_upgrade.py", *argv])
    return cfg, kill_seam


def test_dry_run_mutates_nothing(monkeypatch, tmp_path, capsys):
    """--dry-run must not install, write config, save known-bad, probe, or kill.

    Drives main() through the dry_run branch sites with the mutation seams
    intercepted, so a broken dry_run thread is DETECTED here rather than
    executed against the live ~/.codex/config.toml, the npm global prefix,
    or a possibly-live app-server (plan #2322 kill criterion 4).
    """
    recorder = _RunRecorder(npm_latest="0.999.0", pgrep_pids="4242\n")
    config_text = 'model = "gpt-5.5"\nmodel_reasoning_effort = "xhigh"\n'
    models = [_m("gpt-5.6-sol", 1), _m("gpt-5.5", 7)]
    cfg, kill_seam = _wire_main(monkeypatch, tmp_path, recorder, models, config_text, ["--dry-run"])
    aw_calls: list[tuple[Path, str]] = []

    def fake_atomic_write(path, text):  # mirrors atomic_write(path, text)
        aw_calls.append((path, text))

    monkeypatch.setattr(mod, "atomic_write", fake_atomic_write)
    config_before = cfg.read_bytes()

    rc = mod.main()
    out = capsys.readouterr().out

    assert rc == 0
    # :521 — the CLI upgrade is reported, never installed.
    assert "DRY-RUN would upgrade CLI 0.147.0 -> 0.999.0" in out
    assert recorder.cmds("npm", "install") == []
    # :564 — the current-model probe is skipped; no `codex exec` at all.
    assert "current-model probe" not in out
    assert recorder.cmds("codex", "exec") == []
    # :599-619 — the switch is reported (the listing-backed selection line of
    # acceptance criterion 8), never written; known-bad is never saved.
    assert "DRY-RUN would probe + switch model gpt-5.5 -> gpt-5.6-sol" in out
    assert aw_calls == []
    assert cfg.read_bytes() == config_before
    assert not (tmp_path / "known-bad.json").exists()
    # :439 — main() cannot even reach the kill under --dry-run (no dry-run
    # branch appends to `changed`), so pin the guard directly with
    # live-looking pids: it must return before the inflight re-check + kill.
    assert mod.restart_app_server(dry_run=True) is True
    assert "DRY-RUN would restart pids" in capsys.readouterr().out
    assert kill_seam.call_count == 0


def test_wrapper_alerts_instead_of_silently_exiting_when_project_dir_is_missing(tmp_path):
    """A failed cd must skip the upgrader and reach the rc != 0 alert arm.

    Pre-fix negative control: `cd "$PROJECT_DIR" || exit 1` sat inside the
    brace group (NOT a subshell), so the exit terminated the whole script
    before the alert arm — rc=1, no telegram push, no sentinel, no exit=
    log line. Every assertion below fails under a regression to that shape.
    """
    wrapper_src = (REPO_ROOT / "scripts" / "cron_codex_auto_upgrade.sh").read_text()
    missing = tmp_path / "no-such-project-dir"
    rewritten, n = re.subn(
        r'^PROJECT_DIR="[^"]*"$',
        f'PROJECT_DIR="{missing}"',
        wrapper_src,
        flags=re.MULTILINE,
    )
    assert n == 1, "wrapper PROJECT_DIR line not found — update this fixture"
    wrapper = tmp_path / "cron_codex_auto_upgrade.sh"
    wrapper.write_text(rewritten)

    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("uv", "npm", "codex"):
        stub = bindir / name
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)

    ran_flag = tmp_path / "upgrader-ran.flag"
    upgrader = tmp_path / "fake-upgrader.sh"
    upgrader.write_text(f'#!/bin/sh\ntouch "{ran_flag}"\nexit 0\n')
    upgrader.chmod(0o755)

    telegram_msg = tmp_path / "telegram-msg.txt"
    telegram = tmp_path / "fake-telegram.sh"
    telegram.write_text(f'#!/bin/sh\necho "$1" > "{telegram_msg}"\nexit 0\n')
    telegram.chmod(0o755)

    log_dir = tmp_path / "logs"
    sentinel_dir = tmp_path / "sentinels"
    sidecar = tmp_path / "sidecar.jsonl"
    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env.get('PATH', '')}"
    env["EPS_CODEX_UPGRADE_LOG_DIR"] = str(log_dir)
    env["EPS_CODEX_UPGRADE_SENTINEL_DIR"] = str(sentinel_dir)
    env["EPS_CODEX_UPGRADE_SIDECAR"] = str(sidecar)
    env["EPS_TELEGRAM_PUSH_SCRIPT"] = str(telegram)
    env["EPS_CODEX_UPGRADE_BIN"] = str(upgrader)

    proc = subprocess.run(
        ["bash", str(wrapper)], env=env, capture_output=True, text=True, timeout=60
    )

    assert proc.returncode == 0, proc.stderr  # pre-fix shape exits 1 here
    assert not ran_flag.exists(), "upgrader must be SKIPPED when cd fails"
    logs = list(log_dir.glob("*.log"))
    assert len(logs) == 1
    log_text = logs[0].read_text()
    assert "FATAL: cd" in log_text
    assert "exit=1" in log_text  # the brace group completed with rc=1
    assert telegram_msg.exists(), "failure alert must reach the push script"
    assert "codex_auto_upgrade FAILED (rc=1)" in telegram_msg.read_text()
    assert list(sentinel_dir.glob("failed-*.flag")), "per-day sentinel missing"
    row = json.loads(sidecar.read_text().splitlines()[-1])
    assert row["event"] == "upgrade_failed"
    assert row["rc"] == 1


def test_wrapper_alerts_on_missing_prerequisite_and_unwritable_log_dir(tmp_path):
    """The prerequisite preflight and the log/sentinel-dir mkdir must route
    through the rc != 0 alert arm, never a bare `exit`.

    Pre-fix negative controls (round-2 fix, concern wrapper-prealert-failures):
    (1) the uv/npm/codex preflight ran `exit 1` while LOG_DIR / TELEGRAM_PUSH /
    SIDECAR / SENTINEL were all still undefined, so that exit structurally
    could not alert — wrapper exited 1, telegram never called; (2)
    `mkdir -p "$LOG_DIR" "$SENTINEL_DIR"` was unchecked, so an uncreatable log
    dir failed the brace-group redirect, the group never ran, rc stayed unset,
    and ${rc:-0} read the failure as SUCCESS — exit 0, no telegram, no sidecar
    row. The telegram/sidecar/sentinel assertions below fail under a
    regression to either shape.
    """
    wrapper_src = (REPO_ROOT / "scripts" / "cron_codex_auto_upgrade.sh").read_text()

    ran_flag = tmp_path / "upgrader-ran.flag"
    upgrader = tmp_path / "fake-upgrader.sh"
    upgrader.write_text(f'#!/bin/sh\ntouch "{ran_flag}"\nexit 0\n')
    upgrader.chmod(0o755)

    def run_case(name, wrapper_text, log_dir, sentinel_dir):
        wrapper = tmp_path / f"cron_codex_auto_upgrade_{name}.sh"
        wrapper.write_text(wrapper_text)
        telegram_msg = tmp_path / f"telegram-msg-{name}.txt"
        telegram = tmp_path / f"fake-telegram-{name}.sh"
        telegram.write_text(f'#!/bin/sh\necho "$1" > "{telegram_msg}"\nexit 0\n')
        telegram.chmod(0o755)
        sidecar = tmp_path / f"sidecar-{name}.jsonl"
        env = dict(os.environ)
        env["EPS_CODEX_UPGRADE_LOG_DIR"] = str(log_dir)
        env["EPS_CODEX_UPGRADE_SENTINEL_DIR"] = str(sentinel_dir)
        env["EPS_CODEX_UPGRADE_SIDECAR"] = str(sidecar)
        env["EPS_TELEGRAM_PUSH_SCRIPT"] = str(telegram)
        env["EPS_CODEX_UPGRADE_BIN"] = str(upgrader)
        proc = subprocess.run(
            ["bash", str(wrapper)], env=env, capture_output=True, text=True, timeout=60
        )
        assert proc.returncode == 0, proc.stderr
        assert not ran_flag.exists(), "upgrader must be SKIPPED on a failed setup"
        assert telegram_msg.exists(), "setup failure must reach the push script"
        assert "codex_auto_upgrade FAILED (rc=1)" in telegram_msg.read_text()
        assert list(sentinel_dir.glob("failed-*.flag")), "per-day sentinel missing"
        row = json.loads(sidecar.read_text().splitlines()[-1])
        assert row["event"] == "upgrade_failed"
        assert row["rc"] == 1
        return proc

    # --- Case 1: a missing prerequisite binary must alert. Rewrite the
    # preflight list to include a binary that cannot exist, so the
    # `command -v` miss fires deterministically under the real PATH.
    rewritten, n = re.subn(
        r"^for bin in uv npm codex; do$",
        "for bin in uv npm codex eps-test-missing-prereq-2322; do",
        wrapper_src,
        flags=re.MULTILINE,
    )
    assert n == 1, "wrapper preflight loop not found — update this fixture"
    log_dir = tmp_path / "logs-prereq"
    run_case("prereq", rewritten, log_dir, tmp_path / "sentinels-prereq")
    # The dirs ARE creatable in this case, so the FATAL lands in the log file.
    logs = list(log_dir.glob("*.log"))
    assert len(logs) == 1
    assert "FATAL: eps-test-missing-prereq-2322 not on PATH" in logs[0].read_text()

    # --- Case 2: an uncreatable LOG_DIR (the formerly-unchecked mkdir) must
    # alert. `blocker` is a FILE, so `mkdir -p blocker/logs` fails with
    # ENOTDIR for any uid (a chmod-based block would not bind for root).
    # Verbatim wrapper copy: uv/npm/codex resolve, so ONLY the mkdir fails.
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    proc2 = run_case("mkdir", wrapper_src, blocker / "logs", tmp_path / "sentinels-mkdir")
    # The log file is the unwritable thing, so diagnostics fall back to stderr.
    assert "FATAL" in proc2.stderr and "cannot create log/sentinel dirs" in proc2.stderr
    assert "not on PATH" not in proc2.stderr, "case 2 must isolate the mkdir failure"


def test_broken_current_model_is_excluded_and_replaced(monkeypatch, tmp_path, capsys):
    """A failing current-model probe is recorded into known_bad BEFORE
    candidate_models filters (:569-574 then :583), keyed on this CLI version
    (:294-296), so the broken slug is excluded rather than re-ranked best and
    the config write lands on the next probe-clean slug — never back on the
    broken one (plan #2322 kill criteria 1 + 3)."""
    recorder = _RunRecorder(probe_results={"broken-model": False, "good-model": True})
    config_text = 'model = "broken-model"\nmodel_reasoning_effort = "xhigh"\n'
    models = [_m("broken-model", 1), _m("good-model", 2)]
    cfg, kill_seam = _wire_main(monkeypatch, tmp_path, recorder, models, config_text, [])

    rc = mod.main()
    out = capsys.readouterr().out

    assert rc == 0
    # The failing probe was recorded keyed on the CURRENT CLI version...
    known_bad = json.loads((tmp_path / "known-bad.json").read_text())
    assert known_bad["broken-model"]["cli_version"] == "0.147.0"
    # ...so the broken slug is excluded (never re-probed as a candidate) and
    # the DESTINATION of the config write is the next probe-clean slug.
    assert recorder.probed_slugs() == ["broken-model", "good-model"]
    final = cfg.read_text()
    assert mod.read_config_model(final) == "good-model"
    assert "broken-model" not in final
    assert 'model_reasoning_effort = "xhigh"' in final  # never touched
    assert "model broken-model -> good-model" in out
    # pgrep reported nothing to restart, so the kill seam stays untouched.
    assert kill_seam.call_count == 0
